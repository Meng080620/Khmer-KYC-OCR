# Khmer KYC OCR

Offline, on-prem OCR for the front of the Cambodian national ID card. You send a photo and get back structured, validated fields with a per-field confidence, plus a `needs_review` flag for anything a human should check.

```
photo ─▶ decode (in memory) ─▶ size check ─▶ find + flatten card ─▶ blur check
      ─▶ orient (upright vs 180°, decided by the ID-number format)
      ─▶ crop fixed field boxes ─▶ Kiri OCR per box ─▶ parse + validate ─▶ review decision
```

## Status: not production-ready yet

- **Field boxes are uncalibrated.** `pipeline/template.py` approximates the card layout. Tune it on real cards with `spike/calibrate.py`. Every response carries `template_version`, which ends in `-uncalibrated` until that's done.
- **Accuracy is unmeasured.** Run `spike/bench.py` on 30 or more real, labelled cards. The gate is an ID-number exact match of at least 95% and p95 latency under 1.5s on the target CPU.
- **ID-number format is assumed.** `parse.ID_NUMBER_RE` expects 9 digits. Confirm against the sample set.
- **Model is pinned to the 2026-01-30 Kiri weights.** The newer upload can't be loaded by any `kiri-ocr` release; see the comment on `MODEL_REVISION` in `pipeline/kiri.py`.

## Why not PaddleOCR / Tesseract

- **PaddleOCR** (PP-OCRv5) supports 106 languages, and Khmer is not one of them. `lang='kh'` does not exist.
- **Tesseract** handles Khmer subscript stacking poorly.
- **Kiri OCR** (Apache-2.0, ~23M params) is trained on Khmer and English text lines and runs on CPU.
- **Why not a vision-language model:** it runs offline only if you host it, and it can hallucinate a plausible ID number. Recognition plus strict per-field validation fails loudly instead.

## API

`POST /v1/id-card` takes the raw image as the request body (`Content-Type: image/jpeg`, `image/png` or `image/webp`) and the header `X-API-Token`.

It is deliberately not multipart. Starlette spools multipart files over 1 MB to a temp file, which would put ID images on disk and accept any upload size before the limit check. The raw body is read into memory with the limit enforced as bytes arrive.

| Status | Meaning |
|---|---|
| 200 | `IdCardResult`: `fields.{id_number,name_khmer,name_latin,date_of_birth,sex,expiry_date}` each `{value, raw, confidence, valid}`, plus `needs_review` and `review_reasons` |
| 401 | missing or wrong token (checked before the body is read) |
| 413 | upload over `KYC_OCR_MAX_UPLOAD_BYTES` (default 5 MB) |
| 415 | body is not a JPEG, PNG or WebP content type |
| 422 | `{reason: undecodable \| too_small \| card_not_found \| blurry, detail}`: ask the user to retake the photo |

`GET /healthz` needs no auth.

Review reasons name the field and the problem, never the value. For example, `name_khmer: low confidence 0.62` or `expiry_date: card expired`.

## Run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,kiri]"
```

```bash
.venv/bin/pip install --no-deps "kiri-ocr==0.2.15"
```

`--no-deps` is deliberate. The package's declared dependencies pull in `onnxruntime-gpu` (no macOS build), `datasets`, `matplotlib` and non-headless OpenCV, none of which recognition needs.

```bash
KYC_OCR_API_TOKEN=change-me .venv/bin/uvicorn khmer_kyc_ocr.api:create_app --factory --port 8000
```

```bash
curl -H "X-API-Token: change-me" -H "Content-Type: image/jpeg" --data-binary @card.jpg http://127.0.0.1:8000/v1/id-card
```

Docker (weights baked in, no network at runtime): see the header of the `Dockerfile`.

| Env var | Default | |
|---|---|---|
| `KYC_OCR_API_TOKEN` | required | shared secret for the internal caller |
| `KYC_OCR_MIN_CONFIDENCE` | 0.85 | below this, a field is flagged for review. **Must be calibrated.** Kiri scores clean, correct lines at about 0.44, so the default sends everything to review until `bench.py` gives a real value |
| `KYC_OCR_MIN_SHARPNESS` | 60 | Laplacian variance on the flattened card; calibrate on real photos |
| `KYC_OCR_MIN_SHORT_SIDE_PX` | 600 | |
| `KYC_OCR_MAX_UPLOAD_BYTES` | 5242880 | |
| `KYC_OCR_MAX_CONCURRENT_READS` | 2 | OCR calls in flight per worker; extra requests queue |

## Test

```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy src tests spike
```

`tests/test_kiri_integration.py` runs the real model through the full pipeline on a rendered, upside-down card. It skips when `kiri-ocr` or the cached weights are absent.

The tests need neither torch nor real IDs:
- Cards are synthetic.
- The recognizer is a fake that reads a code painted into each field box.
- `test_concurrent_requests_never_mix_results` checks that parallel requests each get back only their own ID.

## Handling personal data

- Images are decoded in memory and never written to disk.
- Logs record the outcome and timing only.
- Real sample images belong in `spike/samples/`, which is gitignored along with every image type.
- The service must sit on an internal network behind the caller (for example Laravel). It is not an internet-facing endpoint.

## Scaling

- One CPU worker handles roughly 1–3 cards/s, since it makes 6–7 recognizer calls per card. Measure the real rate with `bench.py`.
- To scale, raise `WORKERS` to match the core count (each worker holds its own model copy, about 300 MB), then add replicas behind a load balancer, then move to a GPU with batched field crops.
- Callers should queue work (async jobs) rather than hold user requests open.

## Out of scope for v1

- Back of the card / MRZ
- Face match against a selfie, and liveness checks
- Client apps and backend integration
