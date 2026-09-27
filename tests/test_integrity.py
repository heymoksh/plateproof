"""Pixel-level integrity checks: embedded preview, JPEG quality, size hints, ELA."""

import io

import pytest
from PIL import Image, ImageEnhance

from app import risk_engine
from app.checks import integrity
from tests import factories as f
from tests.conftest import make_ctx


def run(data, filename="photo.jpg", **claim):
    out = integrity.run(make_ctx(data, filename, **claim))
    risk_engine.apply_rules(out.evidence)
    return out


def rules(out):
    return {e.rule for e in out.evidence}


def with_original_thumbnail(original, edited, **exif):
    return f.jpeg(edited, f.camera_exif(thumbnail=f.thumbnail_of(original), **exif))


def letterboxed_thumb(img, box=(160, 120)):
    t = f.thumbnail_of(img, box)
    canvas = Image.new("RGB", box)
    canvas.paste(t, ((box[0] - t.width) // 2, (box[1] - t.height) // 2))
    return canvas


def paste_patch(img, fraction=0.05, seed=7):
    w, h = img.size
    pw, ph = int(w * fraction ** 0.5), int(h * fraction ** 0.5)
    out = img.copy()
    out.paste(f.textured(pw, ph, seed=seed + 999), (w // 3, h // 3))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_genuine_preview_matches(seed):
    out = run(f.camera_photo(seed=seed))
    assert "INFO_THUMBNAIL_MATCH" in rules(out)
    assert out.verifiable_basis


@pytest.mark.parametrize("size", [(800, 450), (360, 640), (480, 640)])
def test_genuine_letterboxed_preview_matches(size):
    img = f.textured(*size, seed=3)
    data = f.jpeg(img, f.camera_exif(thumbnail=letterboxed_thumb(img)))
    assert "INFO_THUMBNAIL_MATCH" in rules(run(data))


def test_local_edit_after_capture_detected():
    original = f.textured(seed=11)
    data = with_original_thumbnail(original, paste_patch(original, 0.05))
    out = run(data)
    assert "THUMBNAIL_MISMATCH" in rules(out)
    assert out.data["embedded_preview"]["local_difference"] > integrity.THUMBNAIL_LOCAL_DIFF_THRESHOLD


def test_crop_to_different_shape_detected():
    original = f.textured(seed=12)
    data = with_original_thumbnail(original, original.crop((0, 60, 640, 420)))
    assert "THUMBNAIL_ASPECT_MISMATCH" in rules(run(data))


def test_global_brightness_edit_is_not_flagged():
    original = f.textured(seed=13)
    data = with_original_thumbnail(original, ImageEnhance.Brightness(original).enhance(1.1))
    assert "THUMBNAIL_MISMATCH" not in rules(run(data))


def test_no_preview_with_camera_metadata_adds_limitation():
    out = run(f.jpeg(f.textured(), f.camera_exif()))
    assert out.data["embedded_preview"]["present"] is False
    assert any("preview" in lim for lim in out.limitations)


@pytest.mark.parametrize("quality", [40, 60, 75, 90])
def test_jpeg_quality_estimate(quality):
    buf = io.BytesIO()
    f.textured().save(buf, "JPEG", quality=quality)
    assert abs(integrity.estimate_jpeg_quality(Image.open(buf)) - quality) <= 2


def test_heavy_compression_adds_limitation():
    out = run(f.jpeg(f.textured(), quality=45))
    assert any("compression" in lim for lim in out.limitations)


def test_screenshot_sized_png_noted_without_points():
    out = run(f.encode(f.textured(1170, 2532), "PNG"), "shot.png")
    assert "INFO_SCREENSHOT_SIZE" in rules(out)
    assert all(e.points == 0 for e in out.evidence)


def test_ai_typical_dimensions_without_metadata():
    out = run(f.encode(f.textured(1024, 1024), "PNG"), "img.png")
    assert "AI_TYPICAL_DIMENSIONS" in rules(out)


def test_ai_dimensions_ignored_when_camera_metadata_present():
    data = f.jpeg(f.textured(1024, 1024), f.camera_exif())
    assert "AI_TYPICAL_DIMENSIONS" not in rules(run(data))


def test_ela_preview_only_for_jpeg():
    assert run(f.camera_photo()).data["ela_preview"].startswith("data:image/jpeg;base64,")
    assert run(f.encode(f.textured(), "PNG"), "x.png").data["ela_preview"] is None
