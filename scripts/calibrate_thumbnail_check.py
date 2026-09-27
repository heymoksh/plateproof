"""Reproduce how THUMBNAIL_LOCAL_DIFF_THRESHOLD was chosen.

Generates synthetic camera-style images, builds a camera preview for each,
then measures the preview-vs-main local difference for genuine images and
for common edits. Run from the project root:

    python scripts/calibrate_thumbnail_check.py --n 200

Synthetic images are a stand-in for real photos. Re-run this with real
camera originals (and edited copies) before trusting the threshold.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.checks.integrity import THUMBNAIL_LOCAL_DIFF_THRESHOLD, compare_with_thumbnail  # noqa: E402
from tests import factories as f  # noqa: E402


def preview(img: Image.Image, box=(160, 120)) -> Image.Image:
    thumb = f.thumbnail_of(img, box)
    canvas = Image.new("RGB", box)
    canvas.paste(thumb, ((box[0] - thumb.width) // 2, (box[1] - thumb.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=85)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def as_jpeg(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def paste(img: Image.Image, fraction: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = img.size
    pw, ph = int(w * fraction ** 0.5), int(h * fraction ** 0.5)
    out = img.copy()
    out.paste(f.textured(pw, ph, seed=seed + 999), (int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    n = parser.parse_args().n
    results: dict[str, list] = {}
    for s in range(n):
        landscape, wide, portrait = f.textured(seed=s), f.textured(800, 450, seed=s + 500), f.textured(360, 640, seed=s + 900)
        cases = {
            "genuine 4:3": (landscape, landscape),
            "genuine 16:9 (letterboxed preview)": (wide, wide),
            "genuine 9:16 (pillarboxed preview)": (portrait, portrait),
            "brightness +15% (global edit)": (ImageEnhance.Brightness(landscape).enhance(1.15), landscape),
            "pasted region 2%": (paste(landscape, 0.02, s), landscape),
            "pasted region 5%": (paste(landscape, 0.05, s), landscape),
            "cropped to 16:9": (landscape.crop((0, 60, 640, 420)), landscape),
        }
        for name, (main, source) in cases.items():
            results.setdefault(name, []).append(compare_with_thumbnail(as_jpeg(main), preview(source)))

    print(f"threshold = {THUMBNAIL_LOCAL_DIFF_THRESHOLD}, n = {n} per case\n")
    print(f"{'case':38s} {'flagged':>8s}  {'local difference (min / median / max)':>40s}")
    for name, rows in results.items():
        flagged = sum(r[0] != "match" for r in rows) / len(rows) * 100
        scores = np.array([r[1] for r in rows if r[1] is not None])
        spread = f"{scores.min():6.1f} / {np.median(scores):6.1f} / {scores.max():6.1f}" if len(scores) else "shape differs"
        print(f"{name:38s} {flagged:7.1f}%  {spread:>40s}")


if __name__ == "__main__":
    main()
