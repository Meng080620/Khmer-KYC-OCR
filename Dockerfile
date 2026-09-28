# Offline Khmer ID OCR. Model weights are downloaded at BUILD time and baked into the image;
# at runtime HF_HUB_OFFLINE=1 forbids any network fetch, so the container runs air-gapped.
#
#   docker build -t khmer-kyc-ocr .
#   docker run --rm --network none -e KYC_OCR_API_TOKEN=... -p 8000:8000 khmer-kyc-ocr
#   (with --network none the port is unreachable from the host; use it only to prove offline
#    start-up, then run on an internal network for real traffic)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    WORKERS=2

WORKDIR /app

# CPU-only torch wheel first: the default wheel bundles CUDA and is several GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[kiri]" \
 && pip install --no-deps "kiri-ocr==0.2.15"

# Fetch weights now so runtime never needs the internet.
RUN python -c "from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer; KiriRecognizer()"

RUN useradd --system --uid 10001 ocr && chown -R ocr /opt/models
USER ocr
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# One model copy per worker process (~300 MB RAM each). Set WORKERS to the CPU core count.
CMD ["sh", "-c", "exec uvicorn khmer_kyc_ocr.api:create_app --factory --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
