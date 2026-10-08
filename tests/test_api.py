"""End-to-end through HTTP with a fake OCR provider, so it runs without torch or real IDs.

The fake derives every value from the input image (see tests/fakes.py), so the concurrency
test is meaningful: each request must get back exactly the ID it sent.
"""

import base64
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from khmer_kyc_ocr.api import create_app
from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W
from khmer_kyc_ocr.schemas import KYCResult, RejectedImage
from khmer_kyc_ocr.service import KYCPipeline
from tests.fakes import FakeOCRProvider, card_photo, id_for_digit
from tests.synthetic import make_card, photograph

TOKEN = "test-token"


def make_client(provider: FakeOCRProvider | None = None, **overrides: object) -> TestClient:
    settings = Settings.model_validate(
        {
            "api_token": TOKEN,
            "min_sharpness": 5.0,
            "min_brightness": 50.0,
            "max_glare_ratio": 0.25,
            "face_detection": False,
            **overrides,
        }
    )
    pipeline = KYCPipeline(provider or FakeOCRProvider(), settings, today=lambda: date(2026, 9, 28))
    return TestClient(create_app(pipeline=pipeline, settings=settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        yield c


@dataclass(frozen=True)
class Reply:
    """Plain status + body, so tests don't depend on the test client's Response class."""

    status_code: int
    text: str


def _post(
    client: TestClient, data: bytes, *, token: str = TOKEN, content_type: str = "image/jpeg"
) -> Reply:
    r = client.post(
        "/v1/id-card",
        content=data,
        headers={"X-API-Token": token, "Content-Type": content_type},
    )
    return Reply(status_code=r.status_code, text=r.text)


def read_ok(client: TestClient, data: bytes) -> KYCResult:
    r = _post(client, data)
    assert r.status_code == 200, r.text
    return KYCResult.model_validate_json(r.text)


def read_rejected(client: TestClient, data: bytes) -> str:
    r = _post(client, data)
    assert r.status_code == 422, r.text
    return RejectedImage.model_validate_json(r.text).reason


def values(result: KYCResult) -> dict[str, str | None]:
    return {name: f.value for name, f in result.fields.items()}


class TestAuth:
    def test_missing_token_is_401(self, client: TestClient) -> None:
        r = client.post(
            "/v1/id-card", content=card_photo(1), headers={"Content-Type": "image/jpeg"}
        )
        assert r.status_code == 401

    def test_wrong_token_is_401(self, client: TestClient) -> None:
        assert _post(client, card_photo(1), token="nope").status_code == 401

    def test_healthz_is_open(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200


class TestRejections:
    def test_oversized_upload_is_413(self) -> None:
        with make_client(max_upload_bytes=1000) as c:
            assert _post(c, card_photo(1)).status_code == 413

    def test_oversized_chunked_upload_is_413(self) -> None:
        def chunks() -> Iterator[bytes]:
            yield from (b"x" * 600 for _ in range(4))

        with make_client(max_upload_bytes=1000) as c:
            r = c.post(
                "/v1/id-card",
                content=chunks(),
                headers={"X-API-Token": TOKEN, "Content-Type": "image/jpeg"},
            )
        assert r.status_code == 413

    def test_non_image_content_type_is_415(self, client: TestClient) -> None:
        assert _post(client, card_photo(1), content_type="multipart/form-data").status_code == 415

    def test_garbage_bytes_are_undecodable(self, client: TestClient) -> None:
        assert read_rejected(client, b"not an image") == "undecodable"

    def test_tiny_image_is_too_small(self, client: TestClient) -> None:
        ok, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 128, dtype=np.uint8))
        assert ok
        assert read_rejected(client, buf.tobytes()) == "too_small"

    def test_photo_without_card(self, client: TestClient) -> None:
        ok, buf = cv2.imencode(".jpg", np.full((1200, 1600, 3), 30, dtype=np.uint8))
        assert ok
        assert read_rejected(client, buf.tobytes()) == "card_not_found"

    def test_blurry_card(self) -> None:
        img = cv2.imdecode(np.frombuffer(card_photo(1), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        ok, buf = cv2.imencode(".jpg", cv2.GaussianBlur(img, (0, 0), 3))
        assert ok
        with make_client(min_sharpness=60.0) as c:
            assert read_rejected(c, buf.tobytes()) == "blurry"

    def test_dark_card(self) -> None:
        card = np.full((CARD_H, CARD_W, 3), 80, dtype=np.uint8)
        ok, buf = cv2.imencode(
            ".jpg", photograph(card, angle_deg=7.0), [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert ok
        with make_client(min_brightness=150.0) as c:
            assert read_rejected(c, buf.tobytes()) == "dark"

    def test_glared_card(self) -> None:
        card = make_card(seed=3)
        card[:, : CARD_W // 2] = 255
        ok, buf = cv2.imencode(
            ".jpg", photograph(card, angle_deg=7.0), [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert ok
        with make_client(max_glare_ratio=0.25) as c:
            assert read_rejected(c, buf.tobytes()) == "glare"


class TestReading:
    def test_reads_all_fields(self, client: TestClient) -> None:
        result = read_ok(client, card_photo(3))
        assert values(result) == {
            "document_type": "national_id",
            "id_number": "010845613",
            "name_kh": "សុខ ដារា",
            "name_en": "SOK DARA",
            "date_of_birth": "1990-02-01",
            "gender": "M",
            "nationality": "Khmer",
            "place_of_birth": "ភ្នំពេញ",
            "address": "ភ្នំពេញ កម្ពុជា",
            "issue_date": "2020-03-05",
            "expiry_date": "2030-03-04",
        }
        assert not result.needs_review
        assert result.ocr_engine == "fake-ocr"
        assert result.schema_version == "kh-kyc-v1"

    def test_upside_down_photo_is_resolved(self, client: TestClient) -> None:
        result = read_ok(client, card_photo(5, angle=183.0))
        assert result.fields["id_number"].value == "010845615"

    def test_low_confidence_flags_review_without_leaking_values(self) -> None:
        with make_client(FakeOCRProvider(confidence_override=0.5)) as c:
            result = read_ok(c, card_photo(2))
        assert result.needs_review
        reasons = " ".join(result.review_reasons)
        assert "low confidence" in reasons
        assert "010845612" not in reasons and "SOK" not in reasons

    def test_expired_card_flags_review(self) -> None:
        settings = Settings.model_validate(
            {"api_token": TOKEN, "min_sharpness": 5.0, "face_detection": False}
        )
        pipeline = KYCPipeline(FakeOCRProvider(), settings, today=lambda: date(2031, 1, 1))
        with TestClient(create_app(pipeline=pipeline, settings=settings)) as c:
            result = read_ok(c, card_photo(2))
        assert result.needs_review
        assert "expiry_date: card expired" in result.review_reasons

    def test_concurrent_requests_never_mix_results(self, client: TestClient) -> None:
        photos = {k: card_photo(k, angle=3.0 * k) for k in range(1, 9)}

        def run(k: int) -> tuple[int, str | None]:
            return k, read_ok(client, photos[k]).fields["id_number"].value

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, [*photos, *photos]))
        assert all(value == id_for_digit(k) for k, value in results)


class TestBase64Endpoint:
    def test_mobile_endpoint_matches_raw_endpoint(self, client: TestClient) -> None:
        data = card_photo(4)
        raw = read_ok(client, data)

        payload = {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "content_type": "image/jpeg",
        }
        r = client.post("/v1/kyc", json=payload, headers={"X-API-Token": TOKEN})
        assert r.status_code == 200, r.text
        b64 = KYCResult.model_validate_json(r.text)
        assert values(b64) == values(raw)

    def test_mobile_endpoint_rejects_invalid_base64(self, client: TestClient) -> None:
        r = client.post(
            "/v1/kyc", json={"image_base64": "!!not-base64!!"}, headers={"X-API-Token": TOKEN}
        )
        assert r.status_code == 422


class TestFaceDetection:
    def test_default_pipeline_builds_and_runs_face_detector(self) -> None:
        settings = Settings.model_validate({"api_token": TOKEN, "min_sharpness": 5.0})
        pipeline = KYCPipeline(FakeOCRProvider(), settings, today=lambda: date(2026, 9, 28))
        with TestClient(create_app(pipeline=pipeline, settings=settings)) as c:
            result = read_ok(c, card_photo(3))
        assert result.face is not None
        assert result.face.detected is False
