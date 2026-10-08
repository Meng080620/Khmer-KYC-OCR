# Architecture

## System overview

Two deployable components, one clean contract:

- **Mobile app** (Expo + TypeScript): camera + guide, auto/manual capture, photo-library
  upload, review & edit, SecureStore token. It is the only caller in the MVP.
- **OCR service** (FastAPI, Python): document detection, perspective correction, quality
  gates, Khmer + English OCR, layout-agnostic field extraction, validation, face detection.

The phone sends `POST /v1/kyc {image_base64}` over HTTPS with `X-API-Token`; the service
returns `KYCResult` or `422 RejectedImage`. The raw-bytes `/v1/id-card` endpoint is kept
for server-to-server callers.

## The pipeline

| Stage | Module | Responsibility |
|---|---|---|
| Image | `pipeline/decode.py` | decode JPEG/PNG/WebP in memory, apply EXIF orientation |
| DocumentDetector | `pipeline/document.py`, `warp.py` | find the largest quadrilateral, perspective-correct |
| QualityDetector | `pipeline/quality.py` | blur (Laplacian), glare (saturated patches), darkness, resolution |
| ImagePreprocessor | `pipeline/preprocess.py` | denoise + CLAHE contrast (never sharpens past the quality gate) |
| OCR | `pipeline/ocr.py`, `lines.py` | detect text lines (layout-agnostic), recognize Khmer + English |
| FieldExtractor | `pipeline/extract.py` | label + format based mapping to fields |
| Validator | `pipeline/validate.py`, `parse.py` | normalize + validate + cross-field checks |
| KYCResult | `schemas.py` | the public contract |

`service.py` (`KYCPipeline`) orchestrates and is stateless per call.

## Layout-agnostic extraction

No hardcoded field boxes. `lines.detect_text_lines` segments text bands with projection
profiles; `LocalOCRProvider` runs Kiri per line; `RuleBasedFieldExtractor` maps lines to
fields by label (Khmer/English) or unambiguous format (9-digit ID, `DD.MM.YYYY` dates).
Only observed fields are returned. Orientation is resolved by content (180° flip retry).

## OCR provider abstraction

```
OCRProvider (read_lines -> list[OcrLine])
 ├── LocalOCRProvider     offline Kiri (default)
 ├── CloudOCRProvider     HTTPS to a configurable endpoint (opt-in)
 └── AIExtractionProvider text -> fields via an LLM (opt-in, in extract.py)
```

Cloud/AI are not enabled by default — sending identity data to a third party is an
explicit privacy decision. Any AI extractor must still pass the `Validator` (a model can
hallucinate).

## On-device vs server-side

On-device: capture, guide, review. Server-side: detection, perspective, quality gates,
OCR (Kiri is ~300 MB, CPU-bound), extraction, validation, face detection. Future
enhancement: move edge detection on-device with react-native-vision-camera (needs a dev
build).

## Face detection

`pipeline/face.py` uses OpenCV's YuNet (ONNX bundled in `src/khmer_kyc_ocr/data/`),
offline. It returns a portrait crop + bbox, preparing for future selfie-to-ID matching
(not implemented in the MVP; neither is liveness).

## Security model

- In-memory processing; nothing touches disk (raw endpoint enforces size while streaming).
- No PII in logs (outcome + latency only); review reasons name fields, never values.
- HTTPS in transit; token in `expo-secure-store` (Keychain/Keystore, this-device-only).
- Constant-time `X-API-Token` comparison.
- No third-party AI by default.
- OCR ≠ verification.

## Data contract

`KYCResult` (`schemas.py`, mirrored in `mobile/src/api/types.ts`): `fields` (only detected
fields; invalid ones present with `value: null`), `confidence`, `face`, `quality`,
`ocr_engine`, `model_revision`, `schema_version`, `needs_review`, `review_reasons`.
