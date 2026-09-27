"""Basic file and image properties."""

from __future__ import annotations

import io

from PIL import ImageCms

from ..models import CheckStatus, Evidence
from . import AnalysisContext, CheckOutput


def run(ctx: AnalysisContext) -> CheckOutput:
    d = ctx.decoded
    icc = d.image.info.get("icc_profile")
    icc_name = None
    if icc:
        try:
            icc_name = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
        except Exception:
            icc_name = "embedded (unreadable)"

    pixels = d.width * d.height
    data = {
        "format": d.format,
        "width": d.width,
        "height": d.height,
        "display_width": d.rgb.width,
        "display_height": d.rgb.height,
        "megapixels": round(pixels / 1e6, 2),
        "file_size_bytes": len(d.data),
        "bytes_per_pixel": round(len(d.data) / pixels, 3) if pixels else None,
        "color_mode": d.mode,
        "frames": d.frames,
        "has_transparency": d.has_alpha,
        "icc_profile": icc_name,
        "notes": d.notes,
    }

    evidence = []
    if d.format_mismatch:
        evidence.append(
            Evidence(
                rule="FORMAT_MISMATCH",
                category="file",
                title="File type does not match its name or declared type",
                detail=d.format_mismatch,
                caveat="Often harmless (files renamed by apps or users), but worth noting.",
            )
        )
    return CheckOutput(CheckStatus.COMPLETED, f"{d.format}, {d.width}x{d.height}", evidence=evidence, data=data)
