# Testing

## Automated tests

Backend uses synthetic cards + a fake OCR provider (no torch, no real IDs, no network):

```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy src tests spike
cd mobile && npx tsc --noEmit && npx expo-doctor
```

Suites: `test_parse.py` (parser edge cases), `test_extract.py` (layout-agnostic mapping),
`test_lines.py` (line detection), `test_quality.py` (blur/glare/dark), `test_face.py`
(offline detector), `test_api.py` (auth, limits, rejections, full read, upside-down,
concurrency, base64 endpoint), `test_kiri_integration.py` (real model end-to-end).

## Real-photo accuracy gate

1. Put 30+ consented card photos in `spike/samples/`.
2. `spike/bench.py --init spike/samples/` then fill `labels.csv`.
3. `spike/bench.py spike/samples/` — prints per-field exact/CER, latency, confidence split.

**Gate:** ID exact ≥ 95% and p95 < 1.5 s. Exit 1 if it fails.

## Required image matrix

clear, blurry, dark, glare, different backgrounds, Khmer-only, Khmer+English, rotated
(including 180°), partially obscured. For each: confirm rejection reason or per-field
accuracy, and fix failures in `extract.py`/`parse.py` (not a template — there is none).

## Not proven by these tests

Document authenticity and identity verification are not tested (out of MVP scope). Face
tests cover loading + no-face only; positive detection needs real images.
