import base64
import binascii
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from khmer_kyc_ocr.config import Settings, get_settings
from khmer_kyc_ocr.pipeline.kiri import MODEL_REVISION
from khmer_kyc_ocr.pipeline.warp import Frame
from khmer_kyc_ocr.schemas import KYCResult, PassportResult, RejectedImage
from khmer_kyc_ocr.service import KYCPipeline

ACCEPTED_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class KYCRequest(BaseModel):
    """Mobile-facing request body: the document image as base64.

    Kept as JSON (not raw bytes) so the React Native client — where sending a raw binary
    ``fetch`` body is unreliable across platforms — has a dependable transport. The image
    is still decoded and processed entirely in memory; nothing touches disk.
    """

    image_base64: str
    content_type: str = "image/jpeg"


def build_ocr_provider(cfg: Settings) -> Any:
    """Construct the OCR provider from settings. Local is the default and runs offline."""
    if cfg.ocr_provider == "cloud":
        from khmer_kyc_ocr.pipeline.ocr import CloudOCRProvider

        if not cfg.cloud_url or cfg.cloud_token is None:
            raise RuntimeError(
                "ocr_provider=cloud requires KYC_OCR_CLOUD_URL and KYC_OCR_CLOUD_TOKEN"
            )
        return CloudOCRProvider(cfg.cloud_url, cfg.cloud_token.get_secret_value())

    from khmer_kyc_ocr.pipeline.ocr import LocalOCRProvider

    if cfg.recognizer == "netra":
        from khmer_kyc_ocr.pipeline.netra import NetraOCRProvider  # heavy: torch, lazy

        return NetraOCRProvider()

    from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer  # heavy: torch, imported lazily

    return LocalOCRProvider(KiriRecognizer(), model_revision=MODEL_REVISION)


def create_app(pipeline: KYCPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests inject a pipeline with a fake OCR provider; production loads Kiri
    once per worker process at startup, so the first request doesn't pay the model-load cost."""
    cfg = settings or get_settings()
    state: dict[str, KYCPipeline] = {}
    # OCR is CPU-bound and torch already uses several threads per call; letting the default
    # 40-thread pool run 40 reads at once just thrashes the CPU and makes every request slow.
    ocr_slots = anyio.CapacityLimiter(cfg.max_concurrent_reads)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["pipeline"] = pipeline or KYCPipeline(build_ocr_provider(cfg), cfg)
        yield
        state.clear()

    app = FastAPI(title="Khmer KYC OCR", version="0.2.0", lifespan=lifespan)

    def require_token(x_api_token: Annotated[str | None, Header()] = None) -> None:
        expected = cfg.api_token.get_secret_value()
        # Constant-time compare: a plain == leaks how many leading characters matched.
        if x_api_token is None or not hmac.compare_digest(x_api_token, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Token")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok" if "pipeline" in state else "loading"}

    async def read_limited_body(request: Request) -> bytes:
        """Read the raw image into memory, enforcing the size limit as bytes arrive.

        Deliberately not multipart: Starlette spools multipart files over 1 MB to a temp file,
        which would put ID images on disk and accept arbitrarily large uploads before any check.
        """
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > cfg.max_upload_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Image too large")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > cfg.max_upload_bytes:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Image too large")
        return bytes(body)

    async def run(data: bytes, frame: Frame | None = None) -> KYCResult | JSONResponse:
        result = await anyio.to_thread.run_sync(
            state["pipeline"].read, data, frame, limiter=ocr_slots
        )
        if isinstance(result, RejectedImage):
            return JSONResponse(status_code=422, content=result.model_dump())
        return result

    @app.post(
        "/v1/kyc",
        response_model=KYCResult,
        dependencies=[Depends(require_token)],
        responses={
            401: {"description": "Bad token"},
            413: {"description": "Upload too large"},
            415: {"description": "Body is not a supported image type"},
            422: {"model": RejectedImage, "description": "Photo unusable; ask for a retake"},
        },
    )
    async def read_kyc(body: KYCRequest) -> KYCResult | JSONResponse:
        if body.content_type.split(";")[0].strip().lower() not in ACCEPTED_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported image type")
        try:
            data = base64.b64decode(body.image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "image_base64 is not valid base64"
            ) from None
        if not data:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "image_base64 is empty")
        return await run(data)

    @app.post(
        "/v1/id-card",
        response_model=KYCResult,
        dependencies=[Depends(require_token)],  # runs before the body is read
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    t: {"schema": {"type": "string", "format": "binary"}} for t in ACCEPTED_TYPES
                },
            }
        },
        responses={
            401: {"description": "Bad token"},
            413: {"description": "Upload too large"},
            415: {"description": "Body is not image/jpeg, image/png or image/webp"},
            422: {"model": RejectedImage, "description": "Photo unusable; ask for a retake"},
        },
    )
    async def read_id_card(
        request: Request,
        content_type: Annotated[str, Header()] = "",
        frame: Annotated[
            str | None,
            Query(
                description=(
                    "Where the app's guide frame was, as 'x,y,width,height' in fractions of "
                    "the photo (0-1). Used only when the card's edges cannot be found."
                ),
                examples=["0.1,0.37,0.8,0.25"],
            ),
        ] = None,
    ) -> KYCResult | JSONResponse:
        if content_type.split(";")[0].strip().lower() not in ACCEPTED_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send the raw image bytes")
        where = parse_frame(frame)
        data = await read_limited_body(request)
        return await run(data, where)

    @app.post(
        "/v1/passport",
        response_model=PassportResult,
        dependencies=[Depends(require_token)],  # runs before the body is read
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    t: {"schema": {"type": "string", "format": "binary"}} for t in ACCEPTED_TYPES
                },
            }
        },
        responses={
            401: {"description": "Bad token"},
            413: {"description": "Upload too large"},
            415: {"description": "Body is not image/jpeg, image/png or image/webp"},
            422: {"model": RejectedImage, "description": "Photo unusable; ask for a retake"},
        },
    )
    async def read_passport(
        request: Request,
        content_type: Annotated[str, Header()] = "",
        frame: Annotated[
            str | None,
            Query(
                description=(
                    "Where the app's guide frame was, as 'x,y,width,height' in fractions of "
                    "the photo (0-1). The passport's photo page is read from that part."
                ),
                examples=["0.1,0.3,0.8,0.42"],
            ),
        ] = None,
    ) -> PassportResult | JSONResponse:
        if content_type.split(";")[0].strip().lower() not in ACCEPTED_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send the raw image bytes")
        where = parse_frame(frame)
        data = await read_limited_body(request)
        result = await anyio.to_thread.run_sync(
            state["pipeline"].read_passport, data, where, limiter=ocr_slots
        )
        if isinstance(result, RejectedImage):
            return JSONResponse(status_code=422, content=result.model_dump())
        return result

    return app


def parse_frame(value: str | None) -> Frame | None:
    """'x,y,width,height' in fractions of the photo, or 422 if it is not a box inside it."""
    if value is None or not value.strip():
        return None
    try:
        x, y, w, h = (float(part) for part in value.split(","))
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "frame must be four numbers: x,y,width,height"
        ) from None
    inside = 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
    if not inside or x + w > 1.0001 or y + h > 1.0001:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "frame must be a box inside the photo (0-1)"
        )
    return (x, y, w, h)
