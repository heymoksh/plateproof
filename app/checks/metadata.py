"""EXIF, XMP, PNG text and content-credential (C2PA) analysis.

Design principles:
- Missing metadata is reported as context, never as a fraud indicator:
  messaging apps, social media and screenshots strip it routinely.
- Metadata that is present can be forged or removed, so every indicator
  carries a caveat.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from PIL import ExifTags

from ..models import CheckStatus, Evidence, ImageSource, Severity
from . import AnalysisContext, CheckOutput

# Names of AI image generators / pipelines as they appear in metadata.
_AI_GENERATOR_PATTERNS = [
    r"stable[\s_-]?diffusion", r"\bsdxl\b", r"\bautomatic1111\b", r"\bcomfyui\b", r"\bmidjourney\b",
    r"\bdall[\s·\-]?e\b", r"adobe firefly", r"google imagen", r"\bimagen[\s-]?\d", r"\bnovelai\b", r"\bleonardo\.ai\b",
    r"\bideogram\b", r"\bblack forest labs\b", r"\bflux\.1\b", r"\binvokeai\b", r"\bfooocus\b",
    r"\bgpt-?image\b", r"\bnano banana\b", r"made with google ai",
]
_AI_RE = re.compile("|".join(_AI_GENERATOR_PATTERNS), re.IGNORECASE)

# IPTC "digital source type" values that declare generative-AI content.
_AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia"}
_SOURCE_TYPE_RE = re.compile(rb"digitalsourcetype/([A-Za-z]+)")

_EDITOR_PATTERNS = [
    r"photoshop", r"lightroom", r"\bgimp\b", r"affinity photo", r"pixelmator", r"paint\.net", r"\bcanva\b",
    r"picsart", r"snapseed", r"facetune", r"\bkrita\b", r"photopea", r"\bfotor\b", r"\bvsco\b", r"airbrush",
    r"meitu", r"photodirector", r"luminar", r"capture one", r"darktable", r"rawtherapee", r"polarr",
]
_EDITOR_RE = re.compile("|".join(_EDITOR_PATTERNS), re.IGNORECASE)

_EXIF_TIME_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M")
_RESAVE_THRESHOLD = timedelta(minutes=10)

# Delivery-window tolerances (clock drift, "delivered" being tapped late).
ORDER_TOLERANCE = timedelta(minutes=5)
DELIVERY_TOLERANCE = timedelta(minutes=15)
LATE_PHOTO_THRESHOLD = timedelta(hours=6)


def _clean(value: Any) -> Any:
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip("\x00 ").strip()
        return text if text.isprintable() else f"<{len(value)} bytes>"
    if isinstance(value, str):
        return value.strip("\x00 ").strip()
    if isinstance(value, tuple):
        return tuple(_clean(v) for v in value)
    try:
        return float(value) if hasattr(value, "numerator") else value
    except (TypeError, ZeroDivisionError, ValueError):
        return str(value)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value or value.startswith("0000"):
        return None
    for fmt in _EXIF_TIME_FORMATS:
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def _gps_decimal(gps: dict) -> dict | None:
    try:
        def to_deg(parts, ref):
            d, m, s = (float(p) for p in parts)
            deg = d + m / 60 + s / 3600
            return -deg if str(ref).upper() in ("S", "W") else deg

        lat = to_deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = to_deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        return {"latitude": round(lat, 6), "longitude": round(lon, 6)}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def read_exif(img) -> tuple[dict, dict, dict]:
    exif = img.getexif()
    base = {ExifTags.TAGS.get(k, str(k)): _clean(v) for k, v in exif.items() if k not in (0x8769, 0x8825)}
    sub = {ExifTags.TAGS.get(k, str(k)): _clean(v) for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()}
    gps = {ExifTags.GPSTAGS.get(k, str(k)): _clean(v) for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
    for noisy in ("MakerNote", "UserComment", "PrintImageMatching", "ComponentsConfiguration"):
        sub.pop(noisy, None)
    return base, sub, gps


def read_xmp(data: bytes) -> dict:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return {}
    end = data.find(b"</x:xmpmeta>", start)
    packet = data[start : end + 12 if end > 0 else start + 200_000][:1_000_000].decode("utf-8", "ignore")

    def values(name: str) -> list[str]:
        found = re.findall(rf'{name}="([^"]*)"|<[^>]*{name}>([^<]*)<', packet)
        return list(dict.fromkeys(v for pair in found for v in pair if v.strip()))

    return {
        "creator_tool": (values("xmp:CreatorTool") or [None])[0],
        "software_agents": values("stEvt:softwareAgent"),
        "digital_source_types": sorted({m.decode() for m in _SOURCE_TYPE_RE.findall(packet.encode())}),
    }


def read_text_chunks(img) -> dict[str, str]:
    """PNG tEXt/iTXt, JPEG comments and similar textual metadata."""
    chunks: dict[str, str] = {}
    for key, value in img.info.items():
        if key in ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp", "dpi", "gamma", "transparency"):
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        if isinstance(value, str) and value.strip():
            chunks[str(key)] = value.strip()
    return chunks


def generator_signatures_in_text(chunks: dict[str, str]) -> list[str]:
    found = []
    params = chunks.get("parameters", "")
    if "Steps:" in params and ("Sampler" in params or "CFG scale" in params):
        found.append("PNG 'parameters' text in the Stable Diffusion WebUI format")
    for key in ("prompt", "workflow"):
        if '"class_type"' in chunks.get(key, ""):
            found.append(f"PNG '{key}' text containing a ComfyUI workflow")
            break
    if any(k in chunks for k in ("invokeai_metadata", "sd-metadata", "Dream")):
        found.append("InvokeAI generation metadata")
    return found


def scan_c2pa(data: bytes) -> dict:
    """Detect (not verify) a C2PA / Content Credentials manifest."""
    idx = data.find(b"c2pa")
    present = idx >= 0 and b"jumb" in data[max(0, idx - 4096) : idx + 4096]
    if not present:
        return {"present": False}
    window = data[idx : idx + 400_000]
    text = window.decode("latin-1")
    source_types = sorted({m.decode() for m in _SOURCE_TYPE_RE.findall(window)} & _AI_SOURCE_TYPES)
    ai_types_bare = sorted(t for t in _AI_SOURCE_TYPES if t in text)
    generators = sorted({m.group(0) for m in _AI_RE.finditer(text)})
    return {
        "present": True,
        "ai_source_types": sorted(set(source_types) | set(ai_types_bare)),
        "generator_mentions": generators,
    }


def run(ctx: AnalysisContext) -> CheckOutput:
    d, claim = ctx.decoded, ctx.claim
    evidence: list[Evidence] = []
    limitations: list[str] = []
    notes: list[str] = []

    try:
        base, sub, gps = read_exif(d.image)
    except Exception:
        base, sub, gps = {}, {}, {}
        notes.append("EXIF block present but unreadable.")

    xmp = read_xmp(d.data)
    chunks = read_text_chunks(d.image)
    c2pa = scan_c2pa(d.data)

    make, model = base.get("Make"), base.get("Model")
    software = base.get("Software")
    captured = _parse_time(sub.get("DateTimeOriginal")) or _parse_time(sub.get("DateTimeDigitized"))
    modified = _parse_time(base.get("DateTime"))
    gps_point = _gps_decimal(gps) if gps else None
    has_camera_metadata = bool(make or model or captured)

    # --- AI generator provenance -------------------------------------------
    ai_hits: list[str] = []
    for label, value in (("EXIF Software", software), ("XMP CreatorTool", xmp.get("creator_tool"))):
        if value and _AI_RE.search(str(value)):
            ai_hits.append(f"{label} = '{value}'")
    for agent in xmp.get("software_agents", []):
        if _AI_RE.search(agent):
            ai_hits.append(f"XMP edit history names '{agent}'")
    ai_types = sorted(set(xmp.get("digital_source_types", [])) & _AI_SOURCE_TYPES)
    if ai_types:
        ai_hits.append(f"XMP digital source type: {', '.join(ai_types)}")
    ai_hits += generator_signatures_in_text(chunks)
    for key, value in chunks.items():
        match = _AI_RE.search(f"{key} {value[:2000]}")
        if match:
            ai_hits.append(f"text metadata '{key}' mentions '{match.group(0)}'")
    if c2pa.get("ai_source_types"):
        ai_hits.append(f"Content Credentials declare: {', '.join(c2pa['ai_source_types'])}")
    elif c2pa.get("generator_mentions"):
        ai_hits.append(f"Content Credentials mention: {', '.join(c2pa['generator_mentions'])}")
    ai_hits = list(dict.fromkeys(ai_hits))

    if ai_hits:
        evidence.append(Evidence(
            rule="AI_PROVENANCE_DECLARED", category="provenance",
            title="Metadata indicates an AI image generator",
            detail="; ".join(ai_hits),
            caveat="Metadata can be added or removed. Its absence says nothing about whether an image is AI-generated.",
        ))

    # --- Editing software --------------------------------------------------
    editors = []
    for value in [software, xmp.get("creator_tool"), *xmp.get("software_agents", [])]:
        if value and _EDITOR_RE.search(str(value)) and not _AI_RE.search(str(value)):
            editors.append(str(value))
    editors = list(dict.fromkeys(editors))
    if editors:
        evidence.append(Evidence(
            rule="EDITING_SOFTWARE", category="metadata",
            title="Saved by image-editing software",
            detail=f"The file was processed with: {', '.join(editors)}. It is not a camera original.",
            caveat="Editing (cropping, brightness) is common and often legitimate. Ask for the original if the edit matters.",
        ))

    # --- Timestamps ----------------------------------------------------------
    now = datetime.now()
    if captured and captured > now + timedelta(days=2):
        evidence.append(Evidence(
            rule="FUTURE_TIMESTAMP", category="metadata",
            title="Capture time is in the future",
            detail=f"The camera timestamp is {captured:%Y-%m-%d %H:%M}.",
            caveat="Usually a wrongly set camera clock, but can also indicate edited metadata.",
        ))
    if captured and captured.year < 1995:
        notes.append(f"Capture year {captured.year} suggests the camera clock was never set.")
    if captured and modified:
        gap = modified - captured
        if gap > _RESAVE_THRESHOLD:
            item = Evidence(
                rule="RESAVED_AFTER_CAPTURE", category="metadata",
                title="File modified after capture",
                detail=f"Captured {captured:%Y-%m-%d %H:%M}, last modified {modified:%Y-%m-%d %H:%M} "
                       f"({_human_delta(gap)} later).",
                caveat="Rotating, cropping or re-exporting a photo also updates this time.",
            )
            if editors:  # already covered by the editing-software indicator
                item.rule, item.severity = "INFO_RESAVED", Severity.INFO
            evidence.append(item)

    # --- Order context: capture time vs order and delivery times ------------
    order_evidence, order_limits = compare_with_order(captured, sub.get("OffsetTimeOriginal"), claim)
    evidence += order_evidence
    limitations += order_limits

    # --- Missing metadata: context, not an indicator --------------------------
    stripping_sources = {ImageSource.MESSAGING_APP, ImageSource.SOCIAL_MEDIA, ImageSource.SCREENSHOT, ImageSource.EMAIL}
    if not has_camera_metadata:
        if claim.image_source == ImageSource.ORIGINAL_CAMERA:
            evidence.append(Evidence(
                rule="SOURCE_METADATA_INCONSISTENT", category="metadata",
                title="Described as an original camera photo, but has no camera metadata",
                detail="Original phone or camera photos normally contain make, model and capture time.",
                caveat="Some apps and privacy settings remove metadata even from originals.",
            ))
        elif claim.image_source in stripping_sources:
            evidence.append(Evidence(
                rule="INFO_NO_METADATA_EXPECTED", category="metadata",
                title="No camera metadata (expected for this source)",
                detail="Messaging apps, social media, email clients and screenshots usually remove camera metadata.",
            ))
        else:
            evidence.append(Evidence(
                rule="INFO_NO_METADATA", category="metadata",
                title="No camera metadata",
                detail="The file has no camera make, model or capture time. This is common for images shared "
                       "through messaging apps or social media, and is not by itself a sign of fraud.",
            ))
        limitations.append(
            "Without camera metadata, capture time, device and editing history cannot be checked."
        )

    if c2pa.get("present") and not c2pa.get("ai_source_types"):
        evidence.append(Evidence(
            rule="INFO_C2PA_PRESENT", category="provenance",
            title="Content Credentials (C2PA) manifest detected",
            detail="The file carries a provenance manifest. This tool detects it but does not verify its signature; "
                   "use a C2PA verifier (e.g. contentcredentials.org/verify) to inspect it.",
        ))

    data = {
        "exif_present": bool(base or sub),
        "camera": {"make": make, "model": model, "lens": sub.get("LensModel")},
        "software": software,
        "timestamps": {
            "captured": captured.isoformat(sep=" ") if captured else None,
            "modified": modified.isoformat(sep=" ") if modified else None,
            "utc_offset": sub.get("OffsetTimeOriginal"),
        },
        "gps": gps_point,
        "xmp": xmp or None,
        "text_metadata": {k: (v[:300] + "…" if len(v) > 300 else v) for k, v in chunks.items()},
        "content_credentials": c2pa,
        "editing_software": editors,
        "ai_generator_signatures": ai_hits,
        "notes": notes,
        "exif": {k: v for k, v in {**base, **sub}.items() if isinstance(v, (str, int, float))},
    }
    device = " ".join(str(v) for v in (make, model) if v)
    if device:
        message = f"Camera metadata found ({device})."
    elif captured:
        message = "Capture time found; camera make and model not recorded."
    else:
        message = "No camera metadata."
    basis_parts = []
    if has_camera_metadata and captured is not None:
        basis_parts.append(f"camera metadata ({device or 'device not recorded'}, captured {captured:%Y-%m-%d %H:%M})")
    if c2pa.get("present"):
        basis_parts.append("a Content Credentials manifest")
    return CheckOutput(
        CheckStatus.COMPLETED, message, evidence=evidence, limitations=limitations, data=data,
        verifiable_basis=bool(basis_parts), basis_note=" and ".join(basis_parts),
    )


def _human_delta(delta: timedelta) -> str:
    days = delta.days
    if days >= 2:
        return f"{days} days"
    hours = delta.total_seconds() / 3600
    return f"{hours:.1f} hours" if hours >= 1 else f"{delta.total_seconds() / 60:.0f} minutes"


def _parse_offset(value: Any) -> timezone | None:
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", str(value or "").strip())
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))


def _align(captured: datetime, offset: timezone | None, other: datetime) -> tuple[datetime, datetime, bool]:
    """Make two timestamps comparable. Returns (captured, other, assumed_local).

    EXIF capture times are local wall-clock times, usually without a zone.
    If both sides carry a zone they are compared exactly; otherwise both are
    treated as local times of the same zone, and the caveat says so.
    """
    if other.tzinfo is not None and offset is not None:
        return captured.replace(tzinfo=offset), other, False
    return captured, other.replace(tzinfo=None), other.tzinfo is not None


def _span(delta: timedelta) -> str:
    minutes = abs(delta.total_seconds()) / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes / 1440:.0f} days"


def compare_with_order(captured: datetime | None, offset_value: Any, claim) -> tuple[list[Evidence], list[str]]:
    """Compare the camera timestamp with the order and delivery times."""
    if not (claim.order_placed_at or claim.delivered_at):
        return [], []
    if captured is None:
        return [], ["The photo's capture time could not be compared with the order times: no camera timestamp."]

    offset = _parse_offset(offset_value)
    caveat = "Phone clocks are usually accurate, but check the time zone and the phone's clock before concluding."
    fmt = "%Y-%m-%d %H:%M"
    evidence: list[Evidence] = []

    if claim.order_placed_at:
        cap, placed, assumed = _align(captured, offset, claim.order_placed_at)
        if cap < placed - ORDER_TOLERANCE:
            evidence.append(Evidence(
                rule="CAPTURE_BEFORE_ORDER", category="order",
                title="Photo was taken before the order was placed",
                detail=f"Captured {cap:{fmt}}, {_span(placed - cap)} before the order was placed ({placed:{fmt}}). "
                       "It cannot show food from this order.",
                caveat=caveat + (" Order time had a time zone; the photo's did not, so local time was assumed." if assumed else ""),
            ))
            return evidence, []

    if claim.delivered_at:
        cap, delivered, assumed = _align(captured, offset, claim.delivered_at)
        extra = " Order time had a time zone; the photo's did not, so local time was assumed." if assumed else ""
        if cap < delivered - DELIVERY_TOLERANCE:
            evidence.append(Evidence(
                rule="CAPTURE_BEFORE_DELIVERY", category="order",
                title="Photo was taken before the order was delivered",
                detail=f"Captured {cap:{fmt}}, {_span(delivered - cap)} before the recorded delivery time ({delivered:{fmt}}).",
                caveat=caveat + " Delivery can also be marked late by the rider." + extra,
            ))
        elif cap > delivered + LATE_PHOTO_THRESHOLD:
            evidence.append(Evidence(
                rule="CAPTURE_LONG_AFTER_DELIVERY", category="order",
                title="Photo taken long after delivery",
                detail=f"Captured {cap:{fmt}}, {_span(cap - delivered)} after delivery ({delivered:{fmt}}).",
                caveat="A late photo can be legitimate, e.g. leftovers checked the next day." + extra,
            ))
        else:
            evidence.append(Evidence(
                rule="INFO_CAPTURE_MATCHES_DELIVERY", category="order",
                title="Capture time fits the delivery window",
                detail=f"Captured {cap:{fmt}}, {_span(cap - delivered)} "
                       f"{'after' if cap >= delivered else 'before'} the recorded delivery ({delivered:{fmt}}).",
            ))
    elif claim.order_placed_at:
        evidence.append(Evidence(
            rule="INFO_CAPTURE_AFTER_ORDER", category="order",
            title="Photo was taken after the order was placed",
            detail="Consistent with the order. Add the delivery time for a tighter check.",
        ))
    return evidence, []
