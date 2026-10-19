from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KYC_OCR_", env_file=".env", extra="ignore")

    # Shared secret for the internal caller (the mobile app / API gateway).
    # Required: the service refuses to start without it.
    api_token: SecretStr
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)

    # --- OCR provider ------------------------------------------------------
    # "local" runs Kiri offline on CPU (default). "cloud" POSTs the flattened card to
    # KYC_OCR_CLOUD_URL over HTTPS — disabled by default because it sends identity data
    # off-network; enabling it is an explicit privacy decision.
    ocr_provider: Literal["local", "cloud"] = "local"
    # Which engine reads each text line when ocr_provider is "local": "kiri" (default) or
    # "netra" (pip install netra-ocr; see pipeline/netra.py for what to weigh first).
    recognizer: Literal["kiri", "netra"] = "kiri"
    cloud_url: str | None = None
    cloud_token: SecretStr | None = None

    # --- Quality gates -----------------------------------------------------
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    min_sharpness: float = Field(default=60.0, ge=0.0)
    min_brightness: float = Field(default=50.0, ge=0.0, le=255.0)
    max_glare_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    min_short_side_px: int = Field(default=600, gt=0)

    # --- Behaviour ----------------------------------------------------------
    face_detection: bool = Field(default=True)
    # Concurrent OCR calls per worker process. Start at 1-2 and raise only if bench shows gain.
    max_concurrent_reads: int = Field(default=2, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
