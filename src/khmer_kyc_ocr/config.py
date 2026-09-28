from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KYC_OCR_", env_file=".env", extra="ignore")

    # Shared secret for the internal caller (e.g. Laravel).
    # Required: the service refuses to start without it.
    api_token: SecretStr
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    # Below this per-field confidence, the result is flagged for human review.
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    # Laplacian variance under this means the photo is too blurry to OCR reliably.
    min_sharpness: float = Field(default=60.0, ge=0.0)
    min_short_side_px: int = Field(default=600, gt=0)
    # Concurrent OCR calls per worker process. Start at 1-2 and raise only if bench shows gain.
    max_concurrent_reads: int = Field(default=2, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
