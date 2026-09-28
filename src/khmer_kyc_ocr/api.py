import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from khmer_kyc_ocr.config import Settings, get_settings
from khmer_kyc_ocr.pipeline.template import TEMPLATE_VERSION
from khmer_kyc_ocr.schemas import IdCardResult, RejectedImage
from khmer_kyc_ocr.service import IdCardReader

ACCEPTED_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


def create_app(reader: IdCardReader | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests inject a reader with a fake recognizer; production loads Kiri once
    per worker process at startup, so the first request doesn't pay the model-load cost."""
    cfg = settings or get_settings()
    state: dict[str, IdCardReader] = {}
    # OCR is CPU-bound and torch already uses several threads per call; letting the default
    # 40-thread pool run 40 reads at once just thrashes the CPU and makes every request slow.
    # Extra requests wait here instead, in arrival order.
    ocr_slots = anyio.CapacityLimiter(cfg.max_concurrent_reads)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if reader is not None:
            state["reader"] = reader
        else:
            from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer  # heavy: torch

            state["reader"] = IdCardReader(KiriRecognizer(), cfg)
        yield
        state.clear()

    app = FastAPI(title="Khmer KYC OCR", version="0.1.0", lifespan=lifespan)

    def require_token(x_api_token: Annotated[str | None, Header()] = None) -> None:
        expected = cfg.api_token.get_secret_value()
        # Constant-time compare: a plain == leaks how many leading characters matched.
        if x_api_token is None or not hmac.compare_digest(x_api_token, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Token")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok" if "reader" in state else "loading", "template": TEMPLATE_VERSION}

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
            if len(body) > cfg.max_upload_bytes:  # Content-Length can lie or be absent
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Image too large")
        return bytes(body)

    @app.post(
        "/v1/id-card",
        response_model=IdCardResult,
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
        request: Request, content_type: Annotated[str, Header()] = ""
    ) -> IdCardResult | JSONResponse:
        if content_type.split(";")[0].strip().lower() not in ACCEPTED_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send the raw image bytes")
        data = await read_limited_body(request)
        # OCR off the event loop: running it inline in `async def` would stall every request.
        result = await anyio.to_thread.run_sync(state["reader"].read, data, limiter=ocr_slots)
        if isinstance(result, RejectedImage):
            return JSONResponse(status_code=422, content=result.model_dump())
        return result

    return app
