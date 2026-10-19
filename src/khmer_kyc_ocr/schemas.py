"""Public KYC data contract shared by the API, pipeline and mobile client.

Design rules:
- ``fields`` contains only fields we actually observed text for. A field that was
  never seen on the document is simply absent — we never invent a value.
- A field that was seen but failed validation is present with ``value=None`` and
  ``valid=False`` so the UI can show the raw text for correction.
- ``raw`` always preserves the recognizer output before normalisation.
"""

from typing import Literal

from pydantic import BaseModel, Field

# The fields the pipeline can currently detect. Keep in sync with parse.PARSERS.
FieldName = Literal[
    "document_type",
    "id_number",
    "name_kh",
    "name_en",
    "date_of_birth",
    "gender",
    "nationality",
    "place_of_birth",
    "address",
    "issue_date",
    "expiry_date",
]


class FieldResult(BaseModel):
    """One extracted field: the normalised value plus the raw text it came from."""

    value: str | None = Field(description="Normalised value, or null when validation failed")
    raw: str = Field(description="Recognizer output before normalisation (preserved for review)")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence for this field")
    valid: bool


class QualityReport(BaseModel):
    """Image quality measured on the flattened card, so a blurred background doesn't reject."""

    short_side_px: int
    sharpness: float = Field(description="Laplacian variance; low means motion/focus blur")
    is_blurry: bool
    brightness: float = Field(ge=0.0, le=255.0, description="Mean luminance of the card")
    is_dark: bool
    glare_ratio: float = Field(ge=0.0, le=1.0, description="Fraction of saturated pixels")
    is_glared: bool
    passed: bool = Field(description="Overall quality gate: image is good enough for OCR")


class FaceResult(BaseModel):
    """The portrait/photo area, if a face was detected on the document."""

    detected: bool
    bbox: tuple[int, int, int, int] | None = Field(
        default=None, description="x, y, w, h on the flattened card"
    )
    image_base64: str | None = Field(
        default=None, description="JPEG-encoded face crop (data URI payload, no prefix)"
    )


class KYCResult(BaseModel):
    """Structured KYC data extracted from an identity document.

    This is *OCR extraction only*. It says nothing about whether the document is
    authentic or the identity is verified — that is a separate, explicit step.
    """

    fields: dict[FieldName, FieldResult]
    confidence: dict[FieldName, float] = Field(
        description="Convenience mirror of each field's confidence"
    )
    face: FaceResult | None = None
    quality: QualityReport
    ocr_engine: str = Field(description="Which OCR provider produced this result")
    model_revision: str | None = Field(default=None, description="Pinned OCR weights, if any")
    schema_version: str = Field(default="kh-kyc-v1")
    needs_review: bool = Field(
        description="True when any field is invalid, low-confidence, or inconsistent"
    )
    review_reasons: list[str] = Field(
        description="Human-readable reasons, naming fields but never field values"
    )
    mrz: list[str] | None = Field(
        default=None,
        description="The card's three machine-readable lines (30 characters each), if read",
    )


class RejectedImage(BaseModel):
    """Returned with 422 when the photo is unusable, so the client can ask for a retake."""

    reason: Literal[
        "undecodable", "too_small", "blurry", "glare", "dark", "card_not_found", "mrz_not_found"
    ]
    detail: str


# What a passport's machine-readable zone carries.
PassportFieldName = Literal[
    "document_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "gender",
    "expiry_date",
]


class PassportResult(BaseModel):
    """What was read from a passport's machine-readable zone.

    Like ``KYCResult`` this is *reading only*: check digits show the zone was read
    correctly, not that the passport is genuine.
    """

    fields: dict[PassportFieldName, FieldResult]
    quality: QualityReport
    ocr_engine: str = Field(description="Which OCR provider produced this result")
    model_revision: str | None = Field(default=None, description="Pinned OCR weights, if any")
    schema_version: str = Field(default="passport-td3-v1")
    needs_review: bool = Field(description="True when any check digit disagrees")
    review_reasons: list[str] = Field(
        description="Human-readable reasons, naming fields but never field values"
    )
