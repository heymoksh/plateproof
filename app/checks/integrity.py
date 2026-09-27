"""Pixel-level integrity checks.

1. Embedded-preview consistency. Cameras store a small preview (thumbnail)
   in EXIF when the photo is taken. Many editors change the main image but
   keep the old preview, so a preview that no longer matches the main image
   is evidence of editing after capture. Global tone changes are normalised
   away (brightness/contrast edits are routine); only local differences or
   a changed shape count.
2. JPEG quality estimate (context: heavy compression limits analysis).
3. Screenshot and AI-generator dimension hints.
4. Error level analysis (ELA) image, returned as a visual aid only. ELA is
   easy to misread, so it is never scored.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import ExifTags, Image, ImageChops

from ..models import CheckStatus, Evidence
from . import AnalysisContext, CheckOutput
from .metadata import read_exif

# Maximum local (8x8-block) difference between tone-normalised preview and
# main image. Chosen from scripts/calibrate_thumbnail_check.py on synthetic
# images: genuine previews stayed below 14; 2% pasted regions mostly exceeded
# 22. Validate on real photos before relying on it.
THUMBNAIL_LOCAL_DIFF_THRESHOLD = 22.0
_COMPARE_SIDE = 48
_DARK_BAR_MAX_MEAN = 24

_SCREEN_SIZES = {
    (1080, 1920), (1080, 2340), (1080, 2400), (1170, 2532), (1179, 2556), (1284, 2778), (1290, 2796),
    (1125, 2436), (1242, 2688), (750, 1334), (1440, 3200), (1440, 3120), (1440, 3088), (1080, 2408),
    (1920, 1080), (2560, 1440), (1366, 768), (1536, 864), (1440, 900), (1280, 720), (2880, 1800),
    (3024, 1964), (2560, 1600), (1600, 900), (1920, 1200), (3840, 2160), (1280, 800), (2560, 1664),
}
_AI_SIZES = {
    (512, 512), (768, 768), (1024, 1024), (2048, 2048), (1024, 1536), (1536, 1024), (1024, 1792),
    (1792, 1024), (832, 1216), (1216, 832), (896, 1152), (1152, 896), (1344, 768), (768, 1344),
    (640, 1536), (1536, 640), (512, 768), (768, 512),
}

# IJG standard luminance quantisation table (quality 50), natural row-major order.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


# --- embedded preview ------------------------------------------------------

def extract_thumbnail(img: Image.Image) -> Image.Image | None:
    raw = img.info.get("exif")
    if not raw:
        return None
    try:
        ifd1 = img.getexif().get_ifd(ExifTags.IFD.IFD1)
    except Exception:
        return None
    offset, length = ifd1.get(0x0201), ifd1.get(0x0202)
    if not offset or not length:
        return None
    base = 6 if raw.startswith(b"Exif\x00\x00") else 0
    blob = raw[base + offset : base + offset + length]
    try:
        thumb = Image.open(io.BytesIO(blob))
        thumb.load()
        return thumb.convert("RGB")
    except Exception:
        return None


def _local_difference(main: Image.Image, thumb: Image.Image) -> float:
    """Max 8x8-block mean absolute difference after tone normalisation."""
    r = thumb.width / thumb.height
    size = (_COMPARE_SIDE, max(8, round(_COMPARE_SIDE / r))) if r >= 1 else (max(8, round(_COMPARE_SIDE * r)), _COMPARE_SIDE)
    a = np.asarray(main.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float64)
    t = np.asarray(thumb.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-6) * t.std() + t.mean()
    diff = np.abs(a - t)
    h, w = diff.shape
    gy, gx = min(8, h), min(8, w)
    bh, bw = h // gy, w // gx
    return float(diff[: bh * gy, : bw * gx].reshape(gy, bh, gx, bw).mean(axis=(1, 3)).max())


def compare_with_thumbnail(main: Image.Image, thumb: Image.Image) -> tuple[str, float | None]:
    """Return ("match" | "mismatch" | "aspect_mismatch", score)."""
    candidates = [main, main.transpose(Image.Transpose.ROTATE_90)]
    best: float | None = None
    W, H = thumb.size
    grey = np.asarray(thumb.convert("L"), dtype=np.float64)
    for m in candidates:
        r = m.width / m.height
        if abs(r / (W / H) - 1) < 0.02:
            score = _local_difference(m, thumb)
            best = score if best is None else min(best, score)
            continue
        # The preview may be letter/pillar-boxed with dark bars. Cameras round
        # the content size differently, so try +/-1 px boxes and keep the best.
        for delta in (-1, 0, 1):
            cw, ch = (W, round(W / r) + delta) if r > W / H else (round(H * r) + delta, H)
            if cw < 16 or ch < 16 or cw > W or ch > H:
                continue
            for ox in {(W - cw) // 2, (W - cw + 1) // 2}:
                for oy in {(H - ch) // 2, (H - ch + 1) // 2}:
                    mask = np.ones_like(grey, dtype=bool)
                    mask[oy : oy + ch, ox : ox + cw] = False
                    if mask.any() and grey[mask].mean() > _DARK_BAR_MAX_MEAN:
                        continue  # the "bars" contain picture content: shapes really differ
                    inner = thumb.crop((ox + 1, oy + 1, ox + cw - 1, oy + ch - 1))
                    mx, my = m.width / cw, m.height / ch
                    sub = m.crop((round(mx), round(my), m.width - round(mx), m.height - round(my)))
                    score = _local_difference(sub, inner)
                    best = score if best is None else min(best, score)
    if best is None:
        return "aspect_mismatch", None
    return ("mismatch" if best > THUMBNAIL_LOCAL_DIFF_THRESHOLD else "match"), round(best, 1)


# --- JPEG quality ----------------------------------------------------------

def estimate_jpeg_quality(img: Image.Image) -> int | None:
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    q = np.asarray(tables[0], dtype=np.float64)
    if q.size != 64:
        return None
    # Pillow returns tables in natural (row-major) order, matching _STD_LUMA.
    scale = float(np.mean(q / _STD_LUMA) * 100)
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(round(min(100, max(1, quality))))


# --- ELA visual aid ---------------------------------------------------------

def ela_preview(rgb: Image.Image, quality: int = 90, max_side: int = 900) -> str:
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    diff = ImageChops.difference(rgb, resaved)
    extreme = max(hi for _, hi in diff.getextrema()) or 1
    diff = diff.point(lambda v: min(255, int(v * 255 / extreme)))
    diff.thumbnail((max_side, max_side))
    out = io.BytesIO()
    diff.save(out, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


# --- check ------------------------------------------------------------------

def run(ctx: AnalysisContext) -> CheckOutput:
    d = ctx.decoded
    evidence: list[Evidence] = []
    limitations: list[str] = []
    data: dict = {}
    basis = False

    try:
        base, sub, _ = read_exif(d.image)
    except Exception:
        base, sub = {}, {}
    has_camera_metadata = bool(base.get("Make") or base.get("Model") or sub.get("DateTimeOriginal"))

    # Embedded preview
    thumb = extract_thumbnail(d.image) if d.format in ("JPEG", "MPO", "TIFF") else None
    if thumb is None:
        data["embedded_preview"] = {"present": False}
        if has_camera_metadata and d.format in ("HEIF", "AVIF"):
            limitations.append(f"The embedded-preview check only supports JPEG files, so it did not run on this "
                               f"{d.format} photo.")
        if has_camera_metadata and d.format in ("JPEG", "MPO"):
            limitations.append("No embedded camera preview was found, so the preview-consistency check could not run.")
    elif min(thumb.size) < 32:
        data["embedded_preview"] = {"present": True, "compared": False, "reason": "too small"}
    else:
        main = d.image.convert("RGB")
        result, score = compare_with_thumbnail(main, thumb)
        data["embedded_preview"] = {"present": True, "compared": True, "result": result, "local_difference": score,
                                    "threshold": THUMBNAIL_LOCAL_DIFF_THRESHOLD, "size": list(thumb.size)}
        # A match only counts as verifiable evidence when camera metadata is also
        # present: apps that strip metadata often regenerate the preview from the
        # current image, so on a stripped file a match proves little. A mismatch
        # is meaningful either way.
        basis = has_camera_metadata and result == "match"
        caveat = ("Some apps rewrite or regenerate previews, and synthetic calibration may not match every camera. "
                  "Treat this as a reason to request the original, not as proof.")
        if result == "aspect_mismatch":
            evidence.append(Evidence(
                rule="THUMBNAIL_ASPECT_MISMATCH", category="integrity",
                title="Main image has a different shape from its embedded preview",
                detail=f"The camera preview is {thumb.width}x{thumb.height} but the main image is {d.width}x{d.height}. "
                       "The main image was probably cropped after capture.",
                caveat=caveat,
            ))
        elif result == "mismatch":
            evidence.append(Evidence(
                rule="THUMBNAIL_MISMATCH", category="integrity",
                title="Embedded preview does not match the main image",
                detail=f"After normalising overall brightness, part of the image differs from the camera's preview "
                       f"(local difference {score} vs threshold {THUMBNAIL_LOCAL_DIFF_THRESHOLD}). "
                       "This is consistent with the picture being edited after it was taken.",
                caveat=caveat,
            ))
        else:
            detail = f"Local difference {score} (threshold {THUMBNAIL_LOCAL_DIFF_THRESHOLD})."
            if not has_camera_metadata:
                detail += (" The file has no camera metadata, so the preview may have been regenerated by the "
                           "app that removed it; this match is weak evidence.")
            evidence.append(Evidence(
                rule="INFO_THUMBNAIL_MATCH", category="integrity",
                title="Embedded preview matches the main image",
                detail=detail,
            ))

    # JPEG compression level
    if d.format in ("JPEG", "MPO"):
        quality = estimate_jpeg_quality(d.image)
        data["jpeg_quality_estimate"] = quality
        if quality is not None and quality < 70:
            limitations.append(f"Heavy JPEG compression (estimated quality {quality}) hides fine detail, "
                               "which is typical after sharing through messaging or social apps.")

    # Size hints (only meaningful without camera metadata)
    size = (d.width, d.height)
    if not has_camera_metadata:
        if d.format == "PNG" and size in _SCREEN_SIZES:
            evidence.append(Evidence(
                rule="INFO_SCREENSHOT_SIZE", category="file",
                title="Dimensions match a common screen resolution",
                detail=f"{d.width}x{d.height} PNG without camera metadata: this may be a screenshot rather than a photo.",
            ))
        if size in _AI_SIZES:
            evidence.append(Evidence(
                rule="AI_TYPICAL_DIMENSIONS", category="file",
                title="Pixel size commonly produced by AI image generators",
                detail=f"{d.width}x{d.height} is a default output size of popular generators, and the file has no camera metadata.",
                caveat="Weak signal: many ordinary images are resized to these sizes.",
            ))

    # ELA visual aid (JPEG only; not scored)
    ela = None
    if d.format in ("JPEG", "MPO"):
        try:
            ela = ela_preview(d.rgb)
        except Exception:
            ela = None
    data["ela_preview"] = ela

    return CheckOutput(CheckStatus.COMPLETED, "Integrity checks completed.", evidence=evidence,
                       limitations=limitations, data=data, verifiable_basis=basis,
                       basis_note="the embedded camera preview matched the main image" if basis else "")
