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
from khmer_kyc_ocr.pipeline.ocr import OcrLine
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, Image
from khmer_kyc_ocr.schemas import KYCResult, PassportResult, RejectedImage
from khmer_kyc_ocr.service import KYCPipeline
from tests.fakes import FakeOCRProvider, card_photo, card_with_digit, id_for_digit
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
    client: TestClient,
    data: bytes,
    *,
    token: str = TOKEN,
    content_type: str = "image/jpeg",
    frame: str | None = None,
) -> Reply:
    r = client.post(
        "/v1/id-card" if frame is None else f"/v1/id-card?frame={frame}",
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


def light_table_photo(digit: int) -> tuple[bytes, str]:
    """A light card on a table of the same shade, and the guide frame around it."""
    photo = np.full((2000, 1500, 3), 235, dtype=np.uint8)
    width, x, y = 1100, 200, 650
    height = round(width * CARD_H / CARD_W)
    photo[y : y + height, x : x + width] = cv2.resize(card_with_digit(digit), (width, height))
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes(), f"{x / 1500:.4f},{y / 2000:.4f},{width / 1500:.4f},{height / 2000:.4f}"


class TestGuideFrame:
    def test_without_a_frame_a_light_card_on_a_light_table_is_not_found(
        self, client: TestClient
    ) -> None:
        data, _ = light_table_photo(4)
        assert read_rejected(client, data) == "card_not_found"

    def test_with_the_frame_it_is_read_and_marked_for_review(self, client: TestClient) -> None:
        data, frame = light_table_photo(4)
        r = _post(client, data, frame=frame)
        assert r.status_code == 200, r.text
        result = KYCResult.model_validate_json(r.text)
        assert result.fields["id_number"].value == id_for_digit(4)
        assert result.needs_review
        assert (
            "card edges not found: read from where the text is or the guide frame"
            in result.review_reasons
        )

    def test_the_frame_is_not_used_when_the_edges_are_found(self, client: TestClient) -> None:
        # The frame points at an empty corner; the card found by its edges must win.
        r = _post(client, card_photo(3), frame="0.0,0.0,0.2,0.2")
        assert r.status_code == 200, r.text
        result = KYCResult.model_validate_json(r.text)
        assert result.fields["id_number"].value == id_for_digit(3)
        assert not result.needs_review

    @pytest.mark.parametrize(
        "frame",
        ["1,2,3", "a,b,c,d", "0.5,0.5,0.6,0.2", "0.1,0.1,0,0.2", "-0.1,0.1,0.5,0.2", "0,0,1.5,1"],
    )
    def test_a_frame_that_is_not_a_box_in_the_photo_is_422(
        self, client: TestClient, frame: str
    ) -> None:
        data, _ = light_table_photo(4)
        assert _post(client, data, frame=frame).status_code == 422

    def test_a_frame_does_not_get_past_the_token(self, client: TestClient) -> None:
        data, frame = light_table_photo(4)
        assert _post(client, data, token="wrong", frame=frame).status_code == 401


class PassportOCR:
    """Returns a passport page's lines for a page with ink on it, nothing for a blank one.
    Upside down (ink in the top half only) reads as garbage, as real OCR would."""

    engine = "fake-ocr"
    model_revision: str | None = None

    def __init__(self, lower: str = "L898902C36UTO7408122F1204159ZE184226B<<<<<10") -> None:
        self.lower = lower

    def read_lines(self, image: Image) -> list[OcrLine]:
        height = image.shape[0]
        dark_top = float(np.mean(image[: height // 2] < 100))
        dark_bottom = float(np.mean(image[height // 2 :] < 100))
        if dark_bottom < 0.01:
            return [] if dark_top < 0.01 else [OcrLine("ƎƧЯƎVƎЯ", (10, 10, 300, 40), 0.3)]
        return [
            OcrLine("PASSPORT", (40, 60, 300, 40), 0.9),
            OcrLine("P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", (40, 800, 1300, 40), 0.9),
            OcrLine(self.lower, (40, 860, 1300, 40), 0.9),
        ]


def passport_photo(*, upside_down: bool = False) -> bytes:
    """A light page with two dark bands where the zone is, sharp enough to pass the gate."""
    page = np.full((1000, 1400, 3), 230, dtype=np.uint8)
    rng = np.random.default_rng(7)
    for y in (800, 860):
        for x in range(40, 1340, 30):
            page[y : y + 40, x : x + int(rng.integers(8, 22))] = 20
    if upside_down:
        page = np.ascontiguousarray(np.rot90(page, 2), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def post_passport(client: TestClient, data: bytes, frame: str | None = None) -> Reply:
    r = client.post(
        "/v1/passport" if frame is None else f"/v1/passport?frame={frame}",
        content=data,
        headers={"X-API-Token": TOKEN, "Content-Type": "image/jpeg"},
    )
    return Reply(status_code=r.status_code, text=r.text)


class TestPassport:
    def test_reads_the_machine_readable_zone(self) -> None:
        with make_client(PassportOCR()) as client:  # type: ignore[arg-type]
            r = post_passport(client, passport_photo())
        assert r.status_code == 200, r.text
        result = PassportResult.model_validate_json(r.text)
        assert {name: f.value for name, f in result.fields.items()} == {
            "document_number": "L898902C3",
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "nationality": "UTO",
            "date_of_birth": "1974-08-12",
            "gender": "F",
            "expiry_date": "2012-04-15",
        }
        assert all(f.valid for f in result.fields.values())
        assert not result.needs_review

    def test_an_upside_down_page_is_turned(self) -> None:
        with make_client(PassportOCR()) as client:  # type: ignore[arg-type]
            r = post_passport(client, passport_photo(upside_down=True))
        assert r.status_code == 200, r.text

    def test_a_failed_check_digit_is_reported_without_the_value(self) -> None:
        wrong = "L898902C36UTO7408132F1204159ZE184226B<<<<<10"
        with make_client(PassportOCR(wrong)) as client:  # type: ignore[arg-type]
            r = post_passport(client, passport_photo())
        result = PassportResult.model_validate_json(r.text)
        assert result.fields["date_of_birth"].value is None
        assert not result.fields["date_of_birth"].valid
        assert result.needs_review
        assert "740813" not in " ".join(result.review_reasons)

    def test_a_page_without_the_zone_is_rejected(self) -> None:
        ok, buf = cv2.imencode(".jpg", np.full((1000, 1400, 3), 230, dtype=np.uint8))
        assert ok
        with make_client(PassportOCR(), min_sharpness=0.0) as client:  # type: ignore[arg-type]
            r = post_passport(client, buf.tobytes())
        assert r.status_code == 422
        assert RejectedImage.model_validate_json(r.text).reason == "mrz_not_found"

    def test_reads_the_framed_part_of_a_larger_photo(self) -> None:
        page = cv2.imdecode(np.frombuffer(passport_photo(), np.uint8), cv2.IMREAD_COLOR)
        assert page is not None
        photo = np.full((3000, 2400, 3), 230, dtype=np.uint8)
        photo[1000:2000, 500:1900] = page
        ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        frame = f"{500 / 2400:.4f},{1000 / 3000:.4f},{1400 / 2400:.4f},{1000 / 3000:.4f}"
        with make_client(PassportOCR()) as client:  # type: ignore[arg-type]
            r = post_passport(client, buf.tobytes(), frame)
        assert r.status_code == 200, r.text

    def test_needs_the_token(self, client: TestClient) -> None:
        r = client.post("/v1/passport", content=b"x", headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 401


class ZoneOCR:
    """A card whose printed text reads badly and whose machine-readable zone reads well."""

    engine = "fake-ocr"
    model_revision: str | None = None

    def __init__(self, second: str = "9002018M3003046KHM<<<<<<<<<<<4") -> None:
        self.second = second

    def read_lines(self, image: Image) -> list[OcrLine]:
        texts = [
            "លេខអត្តសញ្ញាណប័ណ្ណ: 12345",  # too short: fails validation
            "ឈ្មោះជាឡាតាំង: S0K DARA",
            "IDKHM1234567897<<<<<<<<<<<<<<<<<<<",
            self.second,
            "SOK<<DARA<<<<<<<<<<<<<<<<<<<<<<<<<",
        ]
        return [
            OcrLine(text=text, bbox=(10, 10 + i * 60, 900, 40), confidence=0.9, line_number=i + 1)
            for i, text in enumerate(texts)
        ]


class TestMachineReadableZone:
    def test_the_zone_settles_what_the_printed_text_got_wrong(self) -> None:
        with make_client(ZoneOCR()) as client:  # type: ignore[arg-type]
            result = read_ok(client, card_photo(3))
        got = values(result)
        assert got["id_number"] == "123456789"
        assert got["date_of_birth"] == "1990-02-01"
        assert got["expiry_date"] == "2030-03-04"
        assert got["gender"] == "M"
        assert got["name_en"] == "SOK DARA"
        assert result.mrz == [
            "IDKHM1234567897<<<<<<<<<<<<<<<",
            "9002018M3003046KHM<<<<<<<<<<<4",
            "SOK<<DARA<<<<<<<<<<<<<<<<<<<<<",
        ]
        assert not result.needs_review, result.review_reasons

    def test_an_expired_card_is_still_flagged(self) -> None:
        with make_client(ZoneOCR("9002018M2003049KHM<<<<<<<<<<<6")) as client:  # type: ignore[arg-type]
            result = read_ok(client, card_photo(3))
        assert values(result)["expiry_date"] == "2020-03-04"
        assert "expiry_date: card expired" in result.review_reasons

    def test_a_zone_that_fails_its_checks_changes_nothing_and_says_so(self) -> None:
        with make_client(ZoneOCR("9002028M3003046KHM<<<<<<<<<<<4")) as client:  # type: ignore[arg-type]
            result = read_ok(client, card_photo(3))
        assert "date_of_birth" not in result.fields
        assert "machine-readable zone: overall check digit does not agree" in result.review_reasons

    def test_no_zone_means_no_lines_in_the_answer(self, client: TestClient) -> None:
        assert read_ok(client, card_photo(3)).mrz is None
