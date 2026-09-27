"""Independent evidence checks.

Each check receives an AnalysisContext and returns a CheckOutput. Checks never
assign points; the risk engine does that from its rule table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..models import CheckStatus, ClaimContext, Evidence
from ..validation import DecodedImage


@dataclass
class AnalysisContext:
    decoded: DecodedImage
    claim: ClaimContext
    settings: Settings


@dataclass
class CheckOutput:
    status: CheckStatus
    message: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    # True when this check found verifiable provenance (camera metadata,
    # an embedded preview, content credentials), so that "no indicators"
    # can be read as a meaningful result rather than a blank.
    verifiable_basis: bool = False
    # Plain-language description of that basis, used in the report summary.
    basis_note: str = ""
