"""Generate the demo images in samples/ (synthetic, not real photos).

    pip install -r requirements-dev.txt
    python scripts/make_samples.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests import factories as f  # noqa: E402

OUT = ROOT / "samples"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    original = f.textured(800, 600, seed=2026)

    # 1. Camera original: full EXIF, GPS, matching embedded preview.
    camera = f.jpeg(original, f.camera_exif(make="Samsung", model="Galaxy S24", taken="2026:05:02 16:41:09",
                                            thumbnail=f.thumbnail_of(original)), quality=90)
    (OUT / "1_camera_original.jpg").write_bytes(camera)

    # 2. Same photo edited in Photoshop: region replaced, old preview kept.
    edited = original.copy()
    edited.paste(f.textured(200, 150, seed=77), (300, 220))
    (OUT / "2_edited_after_capture.jpg").write_bytes(
        f.jpeg(edited, f.camera_exif(make="Samsung", model="Galaxy S24", taken="2026:05:02 16:41:09",
                                     modified="2026:05:06 21:15:40", software="Adobe Photoshop 26.1",
                                     thumbnail=f.thumbnail_of(original)), quality=90))

    # 3. AI-generated image that declares its generator in PNG metadata.
    generated = f.textured(768, 768, seed=4242)
    (OUT / "3_ai_generated.png").write_bytes(f.png(generated, {
        "parameters": "photo of a dented silver car door, parking lot\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7, Seed: 1"}))

    # 4. The camera original after a messaging app: metadata stripped, resized, recompressed.
    shared = Image.open(io.BytesIO(camera)).convert("RGB").resize((640, 480))
    (OUT / "4_messaging_app_copy.jpg").write_bytes(f.jpeg(shared, quality=55))

    # 5. An old photo, taken weeks before the demo order (placed 2026-05-02 19:00).
    old = f.textured(800, 600, seed=515)
    (OUT / "5_taken_before_order.jpg").write_bytes(
        f.jpeg(old, f.camera_exif(make="Apple", model="iPhone 14", taken="2026:03:12 09:05:00",
                                  thumbnail=f.thumbnail_of(old)), quality=90))
    for p in sorted(OUT.glob("*.*")):
        if p.suffix != ".md":
            print(f"{p.name:32s} {p.stat().st_size / 1024:6.0f} KB")


if __name__ == "__main__":
    main()
