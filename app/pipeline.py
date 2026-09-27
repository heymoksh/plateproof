"""Analysis pipeline: validate once, run independent checks, assess risk.

Each check is isolated: a timeout or exception in one check is recorded as a
failed check and the rest of the analysis continues.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from . import risk_engine
from .checks import AnalysisContext, CheckOutput
from .checks import duplicates, integrity, metadata, ml_detector, properties, vision
from .checks.claim import cross_photo
from .history import HistoryStore
from .config import Settings, get_settings
from .models import CheckResult, CheckStatus, ClaimContext, ClaimReport, RejectedPhoto, Report, ValidationInfo
from .validation import ImageValidationError, validate_and_decode

log = logging.getLogger("plateproof")

DISCLAIMER = (
    "Automated screening of image evidence only. It cannot determine whether a claim is "
    "fraudulent, and it must not be used to approve or reject a claim without human review."
)

# (name, label, function). Order matters only for readability of the report.
CheckFn = Callable[[AnalysisContext], CheckOutput]
LOCAL_CHECKS: list[tuple[str, str, CheckFn]] = [
    ("properties", "File and image properties", properties.run),
    ("metadata", "Metadata and provenance", metadata.run),
    ("integrity", "Image integrity", integrity.run),
    ("duplicates", "Duplicate and reuse detection", duplicates.run),
    ("ml_detector", "Experimental AI-image classifier (advisory)", ml_detector.run),
]


async def _run_check(name: str, label: str, fn: CheckFn, ctx: AnalysisContext, timeout: float) -> CheckOutput:
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, ctx), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("check %s timed out after %ss", name, timeout)
        return CheckOutput(CheckStatus.FAILED, f"Timed out after {timeout:.0f} s.",
                           limitations=[f"{label} did not finish in time, so its evidence is missing."])
    except Exception:
        log.exception("check %s failed", name)
        return CheckOutput(CheckStatus.FAILED, "Unexpected error while running this check.",
                           limitations=[f"{label} failed, so its evidence is missing."])


async def analyze(
    data: bytes,
    filename: str | None,
    content_type: str | None,
    claim: ClaimContext,
    save_to_history: bool = True,
    settings: Settings | None = None,
    use_vision: bool = True,
    case_id: str | None = None,
) -> Report:
    """Analyse one photo. Raises ImageValidationError for unusable uploads."""
    settings = settings or get_settings()
    decoded = await asyncio.to_thread(
        validate_and_decode, data, filename, content_type, settings.max_upload_bytes, settings.max_pixels
    )
    ctx = AnalysisContext(decoded=decoded, claim=claim, settings=settings)
    case_id = case_id or uuid.uuid4().hex[:12].upper()
    analyzed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # The optional vision call runs in parallel with the local checks.
    vision_label = "Vision-model observations (advisory)"
    if use_vision:
        vision_task = asyncio.create_task(
            _run_check("vision", vision_label, vision.run, ctx, settings.vision_timeout_s + 5)
        )
    else:
        vision_task = None

    outputs: dict[str, CheckOutput] = {}
    checks: list[CheckResult] = []
    for name, label, fn in LOCAL_CHECKS:
        out = await _run_check(name, label, fn, ctx, settings.check_timeout_s)
        outputs[name] = out
        checks.append(CheckResult(name=name, label=label, status=out.status, message=out.message))

    if vision_task is not None:
        outputs["vision"] = await vision_task
    else:
        outputs["vision"] = CheckOutput(CheckStatus.SKIPPED, "Turned off for this analysis.", data={"status": "skipped"})
    checks.append(CheckResult(name="vision", label=vision_label,
                              status=outputs["vision"].status, message=outputs["vision"].message))

    evidence = [e for out in outputs.values() for e in out.evidence]
    limitations = [lim for out in outputs.values() for lim in out.limitations]
    risk_engine.apply_rules(evidence)
    basis_notes = [out.basis_note for out in outputs.values() if out.verifiable_basis and out.basis_note]
    basis = any(out.verifiable_basis for out in outputs.values())
    risk = risk_engine.assess(evidence, has_verifiable_basis=basis, basis_notes=basis_notes)

    def section(name: str) -> dict:
        out = outputs.get(name)
        return dict(out.data) if out else {}

    dup = outputs.get("duplicates")
    raw_hashes = dup.data.get("_raw") if dup else None
    if save_to_history and dup and dup.status == CheckStatus.COMPLETED and raw_hashes:
        try:
            await asyncio.to_thread(duplicates.record, ctx, case_id, filename, raw_hashes, analyzed_at)
        except Exception:
            log.exception("could not save submission to history")
            limitations.append("This image could not be saved to the local history, so future uploads "
                               "will not be compared against it.")
    dup_data = section("duplicates")
    dup_data.pop("_raw", None)

    integrity_data = section("integrity")
    ela = integrity_data.pop("ela_preview", None)
    image_info = {**section("properties"), "integrity": integrity_data}

    return Report(
        case_id=case_id,
        analyzed_at=analyzed_at,
        filename=filename,
        claim=claim,
        validation=ValidationInfo(
            passed=True,
            detected_format=decoded.format,
            declared_content_type=decoded.declared_content_type,
            file_extension=decoded.file_extension,
            notes=decoded.notes,
        ),
        risk=risk,
        verified_against=basis_notes,
        evidence=sorted(evidence, key=lambda e: -e.points),
        checks=checks,
        limitations=list(dict.fromkeys(limitations)),
        image=image_info,
        metadata=section("metadata"),
        hashes=dup_data.get("hashes", {}),
        duplicates=dup_data.get("matches", []),
        vision=section("vision"),
        ml_detector=section("ml_detector"),
        ela_preview=ela,
        disclaimer=DISCLAIMER,
    )


async def analyze_claim(
    files: list[tuple[bytes, str | None, str | None]],
    claim: ClaimContext,
    save_to_history: bool = True,
    settings: Settings | None = None,
    use_vision: bool = True,
) -> ClaimReport:
    """Analyse every photo of a refund claim, then compare the photos with each other.

    An unreadable photo in a multi-photo claim is listed as rejected instead of
    failing the whole claim. A single-photo claim raises the validation error.
    """
    settings = settings or get_settings()
    case_id = uuid.uuid4().hex[:12].upper()
    analyzed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    photos: list[Report] = []
    rejected: list[RejectedPhoto] = []
    first_error: ImageValidationError | None = None
    for index, (data, filename, content_type) in enumerate(files, start=1):
        try:
            # History is written after all photos are analysed, so photos in
            # this claim are compared with each other by cross_photo instead.
            photos.append(await analyze(data, filename, content_type, claim, False, settings, use_vision,
                                        case_id=f"{case_id}-P{index}"))
        except ImageValidationError as exc:
            if len(files) == 1:
                raise
            first_error = first_error or exc
            rejected.append(RejectedPhoto(filename=filename, code=exc.code, message=exc.message))
    if not photos:
        raise first_error  # every photo was unusable

    claim_evidence, limitations = cross_photo(photos)
    basis_notes = list(dict.fromkeys(n for p in photos for n in p.verified_against))
    risk = risk_engine.assess_claim(
        [p.risk for p in photos], [p.evidence for p in photos], claim_evidence,
        has_verifiable_basis=bool(basis_notes), basis_notes=basis_notes,
    )
    if rejected:
        limitations.append(f"{len(rejected)} photo(s) could not be analysed and are not included in the result.")

    if save_to_history:
        try:
            await asyncio.to_thread(_record_all, settings, claim, photos, analyzed_at)
        except Exception:
            log.exception("could not save claim photos to history")
            limitations.append("These photos could not be saved to the local history, so future uploads "
                               "will not be compared against them.")

    return ClaimReport(
        case_id=case_id, analyzed_at=analyzed_at, claim=claim, risk=risk,
        claim_evidence=sorted(claim_evidence, key=lambda e: -e.points), photos=photos,
        rejected=rejected, limitations=limitations, disclaimer=DISCLAIMER,
    )


def _record_all(settings: Settings, claim: ClaimContext, photos: list[Report], submitted_at: str) -> None:
    store = HistoryStore(settings.history_db_path)
    for p in photos:
        h = p.hashes
        if not h.get("sha256") or store.exists(h["sha256"], claim.order_id):
            continue
        store.add(case_id=p.case_id, order_id=claim.order_id, customer_id=claim.customer_id,
                  filename=(p.filename or None) and p.filename[:255], sha256=h["sha256"],
                  phash=h["phash"], dhash=h["dhash"], submitted_at=submitted_at)
