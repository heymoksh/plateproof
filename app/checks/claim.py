"""Cross-photo checks for claims with several photos.

Photos of one delivered meal are normally taken on the same phone within a
few minutes. Differences between photos in one claim are worth a look, but
each has innocent explanations, so the points are modest.
"""

from __future__ import annotations

from datetime import datetime

from ..models import Evidence, Report

MAX_SPREAD_HOURS = 3


def _captured(report: Report) -> datetime | None:
    value = (report.metadata.get("timestamps") or {}).get("captured")
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _device(report: Report) -> str | None:
    camera = report.metadata.get("camera") or {}
    name = " ".join(str(v) for v in (camera.get("make"), camera.get("model")) if v)
    return name or None


def cross_photo(photos: list[Report]) -> tuple[list[Evidence], list[str]]:
    if len(photos) < 2:
        return [], []
    evidence: list[Evidence] = []
    limitations: list[str] = []
    label = lambda i: f"photo {i + 1}"  # noqa: E731

    devices = {i: d for i, p in enumerate(photos) if (d := _device(p))}
    if len(set(devices.values())) > 1:
        listing = "; ".join(f"{label(i)}: {d}" for i, d in devices.items())
        evidence.append(Evidence(
            rule="CLAIM_MIXED_DEVICES", category="claim",
            title="Photos come from different cameras",
            detail=f"The camera metadata differs between photos ({listing}).",
            caveat="Two people at the same address can photograph one order, but photos from different "
                   "devices can also mean some were taken from elsewhere.",
        ))

    times = {i: t for i, p in enumerate(photos) if (t := _captured(p))}
    if len(times) >= 2:
        spread = max(times.values()) - min(times.values())
        hours = spread.total_seconds() / 3600
        if hours > MAX_SPREAD_HOURS:
            evidence.append(Evidence(
                rule="CLAIM_TIME_SPREAD", category="claim",
                title="Photos were taken hours apart",
                detail=f"Capture times span {f'{hours / 24:.0f} days' if hours >= 48 else f'{hours:.1f} hours'}. "
                       "Photos of one delivery are usually taken within minutes.",
                caveat="Leftovers or a later discovery can explain a gap.",
            ))
        else:
            evidence.append(Evidence(
                rule="INFO_CLAIM_TIMES_CONSISTENT", category="claim",
                title="Photos were taken close together",
                detail=f"Capture times span {spread.total_seconds() / 60:.0f} minutes.",
            ))

    seen: dict[str, int] = {}
    for i, p in enumerate(photos):
        sha = p.hashes.get("sha256")
        if sha and sha in seen:
            evidence.append(Evidence(
                rule="INFO_CLAIM_DUPLICATE_PHOTO", category="claim",
                title="The same photo was attached twice",
                detail=f"{label(i).capitalize()} is identical to {label(seen[sha])}.",
            ))
        elif sha:
            seen[sha] = i

    with_meta = [i for i, p in enumerate(photos) if _device(p) or _captured(p)]
    if with_meta and len(with_meta) < len(photos):
        evidence.append(Evidence(
            rule="INFO_CLAIM_MIXED_METADATA", category="claim",
            title="Only some photos carry camera metadata",
            detail=f"{len(with_meta)} of {len(photos)} photos have camera metadata. Photos without it may have been "
                   "shared through an app, screenshotted or downloaded.",
        ))
    if len(times) < 2:
        limitations.append("Fewer than two photos have capture times, so their timing could not be compared.")
    return evidence, limitations
