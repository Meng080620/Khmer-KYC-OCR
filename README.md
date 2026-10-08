# Khmer KYC OCR

A mobile application that scans Cambodian identity documents with the phone camera,
extracts Khmer and English text with OCR, and returns structured KYC data with per-field
confidence — plus a review screen where the user corrects anything uncertain before
submitting.

```
Camera  →  Document Detection  →  Auto Crop / Perspective  →  Quality Check
        →  Khmer + English OCR  →  Field Extraction  →  Validation  →  KYC Result
```

> **OCR extraction is not identity verification.** This system reads what is printed on a
> document and flags uncertain fields for human review. It does **not** confirm the document
> is authentic or that the identity is verified — that is a separate, explicit step.

## Repository layout

```
src/khmer_kyc_ocr/        Python FastAPI OCR service (offline, on-prem)
  pipeline/               the modular stages (see docs/ARCHITECTURE.md)
  data/                   bundled YuNet face-detection model (offline)
tests/                    backend tests (synthetic cards, fake + real OCR)
spike/bench.py            go/no-go accuracy benchmark on real, consented photos
mobile/                   Expo + TypeScript app (scanner, review, secure client)
docs/                     ARCHITECTURE.md and TESTING.md
```

## Backend

Python 3.11+ (tested on 3.14). Offline OCR with Kiri OCR (Khmer + English), OpenCV for
document detection, quality gates and face detection.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,kiri]"
.venv/bin/pip install --no-deps "kiri-ocr==0.2.15"
KYC_OCR_API_TOKEN=change-me .venv/bin/uvicorn khmer_kyc_ocr.api:create_app --factory --host 0.0.0.0 --port 8000
```

### API

| Endpoint | Description |
|---|---|
| `POST /v1/kyc` | JSON `{image_base64, content_type}` → `KYCResult` (mobile client) |
| `POST /v1/id-card` | raw image bytes → `KYCResult` (server-to-server) |
| `GET /healthz` | liveness, no auth |

Both OCR endpoints require `X-API-Token`. A poor image returns `422` with a `reason`
(`blurry`, `glare`, `dark`, `too_small`, `card_not_found`, `undecodable`).

### Backend tests

```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy src tests spike
```

## Mobile app

Expo SDK 57 + TypeScript + Expo Router.

```bash
cd mobile
npm install
EXPO_PUBLIC_KYC_API_URL=http://<your-mac-lan-ip>:8000 npm start
```

- `EXPO_PUBLIC_KYC_API_URL` — OCR service URL (a device cannot use `127.0.0.1`).
- `EXPO_PUBLIC_KYC_TOKEN` — dev fallback for the API token.

## Security

Identity data is sensitive:

- Images are decoded and processed **in memory**; never written to disk.
- **No PII is logged** — review reasons name fields, never values.
- Transport is HTTPS; the local provider is offline; token stored in `expo-secure-store`.
- No third-party AI service is called by default (Cloud/AI providers are opt-in).

## Status — MVP scope

In scope: Cambodian ID scanning → Khmer/English OCR → structured extraction → confidence →
user review. Out of scope (by design): back-of-card/MRZ, selfie-to-ID face matching,
liveness, document authenticity, banking/payment integrations. Accuracy must be measured
on real photos with `spike/bench.py` (gate: ID exact ≥ 95%, p95 < 1.5 s). See
docs/TESTING.md and docs/ARCHITECTURE.md.
