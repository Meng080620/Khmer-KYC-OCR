from typing import Literal

from pydantic import BaseModel, Field

FieldName = Literal["id_number", "name_khmer", "name_latin", "date_of_birth", "sex", "expiry_date"]


class FieldResult(BaseModel):
    value: str | None = Field(description="Normalised value, or null when it failed validation")
    raw: str = Field(description="Recognizer output before normalisation")
    confidence: float = Field(ge=0.0, le=1.0)
    valid: bool


class IdCardResult(BaseModel):
    template_version: str
    fields: dict[FieldName, FieldResult]
    needs_review: bool
    review_reasons: list[str]


class RejectedImage(BaseModel):
    """Returned with 422 when the photo is unusable, so the client can ask for a retake."""

    reason: Literal["undecodable", "too_small", "blurry", "card_not_found"]
    detail: str
