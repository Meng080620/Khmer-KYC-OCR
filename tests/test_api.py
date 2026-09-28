"""End-to-end through HTTP with a fake recognizer, so it runs without torch or real IDs.

The fake reads a code painted into each field's box: the mean gray level of the crop tells it
which field (and, for the ID number, which request) it is looking at. That makes the
concurrency test meaningful: each request must get back exactly the ID it sent.
"""

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
from khmer_kyc_ocr.pipeline.recognize import Recognition
from khmer_kyc_ocr.pipeline.template import TEMPLATE_V1
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, Image
from khmer_kyc_ocr.schemas import IdCardResult, RejectedImage
from khmer_kyc_ocr.service import IdCardReader
from tests.synthetic import photograph

TOKEN = "test-token"
ID_LEVELS = {20 + 10 * k: f"01084561{k}" for k in range(8)}  # gray level -> ID number
FIELD_LEVELS = {
    130: Recognition("គោត្តនាម និងនាម: សុខ ដារា", 0.97),
    150: Recognition("SOK DARA", 0.96),
    170: Recognition("ថ្ងៃខែឆ្នាំកំណើត: ០១.០២.១៩៩០", 0.95),
    190: Recognition("ភេទ: ប្រុស", 0.99),
    210: Recognition("សុពលភាព: ០៥.០៣.២០២០ ដល់ថ្ងៃ ០៤.០៣.២០៣០", 0.93),
}
FIELD_LEVEL_BY_NAME = dict(
    zip(
        ("name_khmer", "name_latin", "date_of_birth", "sex", "expiry_date"),
        FIELD_LEVELS,
        strict=True,
    )
)


class FakeRecognizer:
    def __init__(self, confidence_override: float | None = None) -> None:
        self.confidence_override = confidence_override

    def read_line(self, crop: Image) -> Recognition:
        level = float(np.median(crop))  # median ignores border pixels from warp misalignment
        nearest = min([*ID_LEVELS, *FIELD_LEVELS], key=lambda k: abs(k - level))
        if abs(nearest - level) > 4:  # background or the wrong part of the card
            return Recognition("", 0.1)
        if nearest in ID_LEVELS:
            result = Recognition(ID_LEVELS[nearest], 0.98)
        else:
            result = FIELD_LEVELS[nearest]
        if self.confidence_override is not None:
            return Recognition(result.text, self.confidence_override)
        return result


def card_photo(id_digit: int = 0, *, angle: float = 7.0) -> bytes:
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    for name, roi in TEMPLATE_V1.items():
        x0, y0, x1, y1 = roi.to_pixels(CARD_W, CARD_H)
        level = 20 + 10 * id_digit if name == "id_number" else FIELD_LEVEL_BY_NAME[name]
        card[y0:y1, x0:x1] = level
    ok, buf = cv2.imencode(
        ".jpg", photograph(card, angle_deg=angle), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    assert ok
    return buf.tobytes()


def make_client(recognizer: FakeRecognizer | None = None, **overrides: object) -> TestClient:
    settings = Settings.model_validate({"api_token": TOKEN, "min_sharpness": 5.0, **overrides})
    reader = IdCardReader(recognizer or FakeRecognizer(), settings, today=lambda: date(2026, 9, 28))
    return TestClient(create_app(reader=reader, settings=settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        yield c


class TestAuth:
    def test_missing_token_is_401(self, client: TestClient) -> None:
        r = client.post("/v1/id-card", content=card_photo(), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 401

    def test_wrong_token_is_401(self, client: TestClient) -> None:
        assert _post(client, card_photo(), token="nope").status_code == 401

    def test_healthz_is_open(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200


@dataclass(frozen=True)
class Reply:
    """Plain status + body, so tests don't depend on the test client's Response class
    (Starlette's TestClient returns httpx.Response or httpx2.Response depending on install)."""

    status_code: int
    text: str


def _post(
    client: TestClient, data: bytes, *, token: str = TOKEN, content_type: str = "image/jpeg"
) -> Reply:
    # Raw image body, not multipart: multipart spools uploads over 1 MB to a temp file on disk.
    r = client.post(
        "/v1/id-card",
        content=data,
        headers={"X-API-Token": token, "Content-Type": content_type},
    )
    return Reply(status_code=r.status_code, text=r.text)


def read_ok(client: TestClient, data: bytes) -> IdCardResult:
    r = _post(client, data)
    assert r.status_code == 200, r.text
    return IdCardResult.model_validate_json(r.text)


def read_rejected(client: TestClient, data: bytes) -> str:
    r = _post(client, data)
    assert r.status_code == 422, r.text
    return RejectedImage.model_validate_json(r.text).reason


def values(result: IdCardResult) -> dict[str, str | None]:
    return {name: f.value for name, f in result.fields.items()}


class TestRejections:
    def test_oversized_upload_is_413(self) -> None:
        with make_client(max_upload_bytes=1000) as c:
            assert _post(c, card_photo()).status_code == 413

    def test_oversized_chunked_upload_is_413(self) -> None:
        # No Content-Length: the limit must be enforced while streaming, not trusted from headers.
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
        assert _post(client, card_photo(), content_type="multipart/form-data").status_code == 415

    def test_garbage_bytes_are_undecodable(self, client: TestClient) -> None:
        assert read_rejected(client, b"not an image") == "undecodable"

    def test_photo_without_card(self, client: TestClient) -> None:
        ok, buf = cv2.imencode(".jpg", np.full((1200, 1600, 3), 30, dtype=np.uint8))
        assert ok
        assert read_rejected(client, buf.tobytes()) == "card_not_found"

    def test_blurry_card(self) -> None:
        img = cv2.imdecode(np.frombuffer(card_photo(), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        ok, buf = cv2.imencode(".jpg", cv2.GaussianBlur(img, (0, 0), 3))
        assert ok
        with make_client(min_sharpness=60.0) as c:
            assert read_rejected(c, buf.tobytes()) == "blurry"


class TestReading:
    def test_reads_all_fields(self, client: TestClient) -> None:
        result = read_ok(client, card_photo(3))
        assert values(result) == {
            "id_number": "010845613",
            "name_khmer": "សុខ ដារា",
            "name_latin": "SOK DARA",
            "date_of_birth": "1990-02-01",
            "sex": "M",
            "expiry_date": "2030-03-04",
        }
        assert not result.needs_review
        assert result.template_version.endswith("uncalibrated")

    def test_upside_down_photo_is_resolved_by_id_number(self, client: TestClient) -> None:
        result = read_ok(client, card_photo(5, angle=183.0))
        assert result.fields["id_number"].value == "010845615"

    def test_low_confidence_flags_review_without_leaking_values(self) -> None:
        with make_client(FakeRecognizer(confidence_override=0.5)) as c:
            result = read_ok(c, card_photo())
        assert result.needs_review
        reasons = " ".join(result.review_reasons)
        assert "low confidence" in reasons
        assert "010845610" not in reasons and "SOK" not in reasons

    def test_expired_card_flags_review(self) -> None:
        settings = Settings.model_validate({"api_token": TOKEN, "min_sharpness": 5.0})
        reader = IdCardReader(FakeRecognizer(), settings, today=lambda: date(2031, 1, 1))
        with TestClient(create_app(reader=reader, settings=settings)) as c:
            result = read_ok(c, card_photo())
        assert result.needs_review
        assert "expiry_date: card expired" in result.review_reasons

    def test_concurrent_requests_never_mix_results(self, client: TestClient) -> None:
        photos = {k: card_photo(k, angle=3.0 * k) for k in range(8)}

        def run(k: int) -> tuple[int, str | None]:
            return k, read_ok(client, photos[k]).fields["id_number"].value

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, [*photos, *photos]))
        assert all(value == f"01084561{k}" for k, value in results)
