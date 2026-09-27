"""Deterministic, explainable risk assessment.

This is the ONLY place where severities and points are assigned. Checks
report what they observed (by rule id); this module decides how much each
observation counts.

IMPORTANT: the point values and thresholds below are hand-set heuristics.
They have not been learned from or calibrated on labelled claim data, so the
resulting score is a triage aid, not a probability of fraud. To calibrate
them, collect labelled examples and fit/validate thresholds (see README).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Evidence, RiskAssessment, Severity


@dataclass(frozen=True)
class Rule:
    severity: Severity
    points: int
    rationale: str


RULES: dict[str, Rule] = {
    # Strong indicators
    "AI_PROVENANCE_DECLARED": Rule(Severity.HIGH, 60, "File metadata names an AI image generator."),
    "DUPLICATE_EXACT_OTHER_ORDER": Rule(Severity.HIGH, 50, "Identical file already submitted for a different order."),
    "DUPLICATE_NEAR_OTHER_ORDER": Rule(Severity.HIGH, 40, "Visually near-identical image submitted for a different order."),
    "CAPTURE_BEFORE_ORDER": Rule(Severity.HIGH, 50, "Camera timestamp is earlier than the order itself."),
    "CAPTURE_BEFORE_DELIVERY": Rule(Severity.HIGH, 40, "Camera timestamp is earlier than the delivery."),
    # Moderate indicators
    "DUPLICATE_EXACT_UNKNOWN_ORDER": Rule(Severity.MEDIUM, 30, "Identical file seen before; order linkage unknown."),
    "DUPLICATE_NEAR_UNKNOWN_ORDER": Rule(Severity.MEDIUM, 25, "Near-identical image seen before; order linkage unknown."),
    "THUMBNAIL_MISMATCH": Rule(Severity.MEDIUM, 30, "Embedded camera preview differs from the main image."),
    "THUMBNAIL_ASPECT_MISMATCH": Rule(Severity.MEDIUM, 20, "Main image was cropped or reshaped after the preview was created."),
    "EDITING_SOFTWARE": Rule(Severity.MEDIUM, 20, "File was last saved by image-editing software."),
    "FUTURE_TIMESTAMP": Rule(Severity.MEDIUM, 20, "Camera timestamp lies in the future."),
    "CLAIM_MIXED_DEVICES": Rule(Severity.MEDIUM, 20, "Photos in one claim come from different cameras."),
    # Weak indicators
    "CAPTURE_LONG_AFTER_DELIVERY": Rule(Severity.LOW, 10, "Photo taken more than 6 hours after delivery."),
    "RESAVED_AFTER_CAPTURE": Rule(Severity.LOW, 10, "File modified well after it was captured."),
    "CLAIM_TIME_SPREAD": Rule(Severity.LOW, 10, "Photos in one claim were taken hours apart."),
    "SOURCE_METADATA_INCONSISTENT": Rule(Severity.LOW, 10, "Described as an original camera photo but has no camera metadata."),
    "FORMAT_MISMATCH": Rule(Severity.LOW, 5, "File extension or declared type does not match the content."),
    "AI_TYPICAL_DIMENSIONS": Rule(Severity.LOW, 5, "Exact pixel size commonly produced by AI generators, with no camera metadata."),
}

HIGH_SCORE_THRESHOLD = 50
REVIEW_SCORE_THRESHOLD = 20

SCORE_NOTE = (
    "Heuristic score: the sum of hand-set rule points, capped at 100. "
    "It is not a probability of fraud and has not been calibrated on labelled claims."
)

_RECOMMENDATIONS = {
    "HIGH_REVIEW_PRIORITY": (
        "Prioritise manual review by an investigator. Verify each listed indicator "
        "before taking any action on the claim."
    ),
    "REVIEW_RECOMMENDED": (
        "Manual review recommended. If needed, ask the claimant for the original, "
        "unedited photo taken directly from their device."
    ),
    "NO_SIGNIFICANT_INDICATORS": (
        "No image-level indicators were found by the automated checks. Continue normal "
        "claim handling. This does not confirm that the claim is genuine."
    ),
    "INSUFFICIENT_EVIDENCE": (
        "The image carries too little verifiable information to assess (for example, "
        "metadata removed by a messaging app or a screenshot). If verification matters, "
        "request the original photo taken directly from the device."
    ),
}


def apply_rules(evidence: list[Evidence]) -> list[Evidence]:
    """Stamp severity and points onto evidence from the rule table.

    Advisory evidence (vision model, ML detector) and unknown rules stay at
    zero points, whatever the producing check asked for.
    """
    for item in evidence:
        rule = RULES.get(item.rule)
        if rule is None or item.advisory:
            item.points = 0
            if item.severity != Severity.INFO and item.advisory:
                item.severity = Severity.INFO
            continue
        item.severity = rule.severity
        item.points = rule.points
    return evidence


def assess(evidence: list[Evidence], has_verifiable_basis: bool, basis_notes: list[str] | None = None) -> RiskAssessment:
    """Turn scored evidence into a review level.

    `has_verifiable_basis` is True when the file contained enough provenance
    information (camera metadata with timestamps, an embedded preview, or
    content credentials) for "nothing found" to mean something.
    """
    scored = sorted((e for e in evidence if e.points > 0), key=lambda e: -e.points)
    score = min(100, sum(e.points for e in scored))
    count = f"{len(scored)} indicator{'s' if len(scored) != 1 else ''}"
    severities = {e.severity for e in scored}

    if Severity.HIGH in severities or score >= HIGH_SCORE_THRESHOLD:
        level = "HIGH_REVIEW_PRIORITY"
        summary = f"{count} found, including at least one strong indicator." if Severity.HIGH in severities else f"{count} found with a combined score of {score}."
    elif Severity.MEDIUM in severities or score >= REVIEW_SCORE_THRESHOLD:
        level = "REVIEW_RECOMMENDED"
        summary = f"{count} found that {'deserves' if len(scored) == 1 else 'deserve'} a closer look."
    elif has_verifiable_basis:
        level = "NO_SIGNIFICANT_INDICATORS"
        checked = f"Verified against {'; '.join(basis_notes)}. " if basis_notes else ""
        summary = checked + (
            "No significant indicators were found."
            if not scored
            else "Only minor observations were found; none is significant on its own."
        )
    else:
        level = "INSUFFICIENT_EVIDENCE"
        summary = (
            "No indicators were found, but the image lacks the metadata needed to verify it, "
            "so the absence of indicators is not meaningful."
        )

    return RiskAssessment(
        level=level,
        score=score,
        score_note=SCORE_NOTE,
        summary=summary,
        recommendation=_RECOMMENDATIONS[level],
        reasons=[f"{e.title} (+{e.points})" for e in scored],
    )


def assess_claim(photo_risks: list[RiskAssessment], photo_evidence: list[list[Evidence]],
                 claim_evidence: list[Evidence], has_verifiable_basis: bool,
                 basis_notes: list[str] | None = None) -> RiskAssessment:
    """Combine per-photo results and cross-photo evidence into one claim result.

    Score = the highest single-photo score + cross-photo points (capped at 100).
    Photo scores are not summed: sending more photos should not raise the score
    by itself. Severities from every photo still count toward the level.
    """
    apply_rules(claim_evidence)
    if len(photo_risks) == 1 and not any(e.points for e in claim_evidence):
        return photo_risks[0]
    claim_scored = sorted((e for e in claim_evidence if e.points > 0), key=lambda e: -e.points)
    best = max((r.score for r in photo_risks), default=0)
    score = min(100, best + sum(e.points for e in claim_scored))
    severities = {e.severity for ev in photo_evidence for e in ev if e.points > 0} | {e.severity for e in claim_scored}
    n_flagged = sum(1 for r in photo_risks if r.score > 0)

    reasons = [f"Photo {i + 1}: {reason}" for i, r in enumerate(photo_risks) for reason in r.reasons]
    reasons += [f"All photos: {e.title} (+{e.points})" for e in claim_scored]
    if Severity.HIGH in severities or score >= HIGH_SCORE_THRESHOLD:
        level = "HIGH_REVIEW_PRIORITY"
        summary = f"{n_flagged} of {len(photo_risks)} photos have indicators, including at least one strong indicator."
    elif Severity.MEDIUM in severities or score >= REVIEW_SCORE_THRESHOLD:
        level = "REVIEW_RECOMMENDED"
        summary = (f"{n_flagged} of {len(photo_risks)} photos have indicators" if n_flagged else "The photos disagree with each other")
        summary += " that deserve a closer look." if n_flagged else " in ways that deserve a closer look."
    elif has_verifiable_basis:
        level = "NO_SIGNIFICANT_INDICATORS"
        checked = f"Verified against {'; '.join(basis_notes)}. " if basis_notes else ""
        summary = checked + "No significant indicators were found across the photos."
    else:
        level = "INSUFFICIENT_EVIDENCE"
        summary = ("No indicators were found, but none of the photos has the metadata needed to verify it, "
                   "so the absence of indicators is not meaningful.")
    return RiskAssessment(level=level, score=score, score_note=SCORE_NOTE, summary=summary,
                          recommendation=_RECOMMENDATIONS[level], reasons=reasons)
