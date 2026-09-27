"""Data structures shared across the analysis pipeline and the API response."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Severity(str, Enum):
    INFO = "info"      # context only, never adds to the risk score
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CheckStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"          # not applicable (e.g. thumbnail check on a PNG)
    UNAVAILABLE = "unavailable"  # optional feature not configured (no key, no model)
    FAILED = "failed"            # tried and errored; the rest of the analysis continues


class ImageSource(str, Enum):
    UNKNOWN = "unknown"
    ORIGINAL_CAMERA = "original_camera"
    MESSAGING_APP = "messaging_app"
    SOCIAL_MEDIA = "social_media"
    SCREENSHOT = "screenshot"
    EMAIL = "email"
    OTHER = "other"


class Evidence(BaseModel):
    """One observation produced by a check.

    `rule` links the observation to an entry in the risk engine's rule table,
    which is the only place that assigns severity and points.
    """

    rule: str
    category: Literal["file", "metadata", "provenance", "integrity", "reuse", "order", "claim", "vision", "model"]
    title: str
    detail: str
    severity: Severity = Severity.INFO
    points: int = 0
    caveat: str | None = None
    advisory: bool = False  # True for model/LLM output that is shown but never scored


class CheckResult(BaseModel):
    name: str
    label: str
    status: CheckStatus
    message: str = ""


class ComplaintType(str, Enum):
    UNKNOWN = "unknown"
    MISSING_ITEM = "missing_item"
    SPILLED_OR_DAMAGED = "spilled_or_damaged"
    WRONG_ITEM = "wrong_item"
    QUALITY_ISSUE = "quality_issue"          # stale, undercooked, burnt, cold
    FOREIGN_OBJECT = "foreign_object"        # hair, insect, plastic
    OTHER = "other"


class ClaimContext(BaseModel):
    """Optional order details for a refund claim. Every field is optional."""

    order_id: str | None = Field(default=None, max_length=100)
    customer_id: str | None = Field(default=None, max_length=100)
    restaurant: str | None = Field(default=None, max_length=200)
    ordered_items: str | None = Field(default=None, max_length=1000)
    order_placed_at: datetime | None = None
    delivered_at: datetime | None = None
    complaint_type: ComplaintType = ComplaintType.UNKNOWN
    description: str | None = Field(default=None, max_length=2000)
    image_source: ImageSource = ImageSource.UNKNOWN

    @model_validator(mode="after")
    def _delivery_after_order(self):
        if self.order_placed_at and self.delivered_at:
            a, b = self.order_placed_at, self.delivered_at
            if (a.tzinfo is None) == (b.tzinfo is None) and b < a:
                raise ValueError("delivered_at is earlier than order_placed_at")
        return self


RiskLevel = Literal[
    "HIGH_REVIEW_PRIORITY",
    "REVIEW_RECOMMENDED",
    "NO_SIGNIFICANT_INDICATORS",
    "INSUFFICIENT_EVIDENCE",
]


class RiskAssessment(BaseModel):
    level: RiskLevel
    score: int = Field(description="Heuristic risk score 0-100. Not a probability of fraud.")
    score_note: str
    summary: str
    recommendation: str
    reasons: list[str]


class ValidationInfo(BaseModel):
    passed: bool
    detected_format: str
    declared_content_type: str | None
    file_extension: str | None
    notes: list[str] = []


class Report(BaseModel):
    case_id: str
    analyzed_at: str
    filename: str | None
    claim: ClaimContext
    validation: ValidationInfo
    risk: RiskAssessment
    verified_against: list[str] = Field(default=[], description="What verifiable provenance this photo carried.")
    evidence: list[Evidence]
    checks: list[CheckResult]
    limitations: list[str]
    image: dict[str, Any]
    metadata: dict[str, Any]
    hashes: dict[str, Any]
    duplicates: list[dict[str, Any]]
    vision: dict[str, Any] | None
    ml_detector: dict[str, Any]
    ela_preview: str | None = Field(default=None, description="Data URL of the error-level-analysis visual aid (JPEG only).")
    disclaimer: str


class RejectedPhoto(BaseModel):
    filename: str | None
    code: str
    message: str


class ClaimReport(BaseModel):
    """Result for a whole refund claim (one or more photos)."""

    case_id: str
    analyzed_at: str
    claim: ClaimContext
    risk: RiskAssessment
    claim_evidence: list[Evidence] = Field(description="Cross-photo observations for the claim as a whole.")
    photos: list[Report]
    rejected: list[RejectedPhoto] = []
    limitations: list[str] = []
    disclaimer: str
