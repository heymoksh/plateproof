import io

import numpy as np
import pytest
from PIL import Image

from app.validation import HEIF_SUPPORTED, ImageValidationError, sniff_format, to_rgb, validate_and_decode
from tests import factories as f

MB = 1024 * 1024


def decode(data, filename="x.jpg", content_type=None, max_bytes=25 * MB, max_pixels=80_000_000):
    return validate_and_decode(data, filename, content_type, max_bytes, max_pixels)


@pytest.mark.parametrize(
    "fmt,ext",
    [("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp"), ("BMP", ".bmp"), ("TIFF", ".tif"), ("AVIF", ".avif")],
)
def test_common_formats_decode(fmt, ext):
    data = f.encode(f.textured(), fmt)
    d = decode(data, "photo" + ext)
    assert d.format == fmt
    assert d.rgb.mode == "RGB" and d.rgb.size == (640, 480)
    assert d.format_mismatch is None


@pytest.mark.skipif(not HEIF_SUPPORTED, reason="pillow-heif not installed")
def test_heic_decodes():
    data = f.encode(f.textured(), "HEIF")
    d = decode(data, "IMG_0001.HEIC", "image/heic")
    assert d.format == "HEIF" and d.rgb.mode == "RGB"


def test_animated_gif_uses_first_frame():
    frames = [f.textured(seed=i).convert("P") for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    d = decode(buf.getvalue(), "a.gif")
    assert d.frames == 3
    assert any("first frame" in n for n in d.notes)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "L", "1", "CMYK", "P"])
def test_colour_modes_normalise_to_rgb(mode):
    img = f.textured().convert(mode)
    fmt = "JPEG" if mode in ("CMYK", "L") else "PNG"
    d = decode(f.encode(img, fmt), "x." + ("jpg" if fmt == "JPEG" else "png"))
    assert d.rgb.mode == "RGB"


def test_transparency_composited_on_white_not_black():
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 0))  # fully transparent red
    d = decode(f.encode(img, "PNG"), "t.png")
    assert d.rgb.getpixel((10, 10)) == (255, 255, 255)
    assert d.has_alpha


def test_sixteen_bit_grayscale_is_scaled_not_clipped():
    arr = np.linspace(0, 65535, 100 * 100).reshape(100, 100).astype(np.uint16)
    img = Image.fromarray(arr, "I;16")
    rgb = to_rgb(img)
    values = np.asarray(rgb)[..., 0]
    assert values.min() == 0 and values.max() >= 250  # full range preserved


def test_exif_orientation_is_applied():
    img = f.textured(400, 200)
    data = f.jpeg(img, f.camera_exif(orientation=6))  # rotate 90 degrees
    d = decode(data)
    assert (d.width, d.height) == (400, 200)
    assert d.rgb.size == (200, 400)
    assert d.orientation == 6


def test_empty_file_rejected():
    with pytest.raises(ImageValidationError) as e:
        decode(b"")
    assert e.value.code == "empty_file" and e.value.http_status == 400


def test_oversized_file_rejected():
    with pytest.raises(ImageValidationError) as e:
        decode(f.jpeg(f.textured()), max_bytes=1000)
    assert e.value.code == "file_too_large" and e.value.http_status == 413


def test_too_many_pixels_rejected_before_decoding():
    data = f.encode(Image.new("RGB", (3000, 3000)), "PNG")
    with pytest.raises(ImageValidationError) as e:
        decode(data, "big.png", max_pixels=1_000_000)
    assert e.value.code == "too_many_pixels"


@pytest.mark.parametrize(
    "payload,name",
    [
        (b"hello, this is text", "notes.jpg"),
        (b'<svg xmlns="http://www.w3.org/2000/svg"/>', "drawing.svg"),
        (b"%PDF-1.7 fake", "claim.pdf"),
    ],
)
def test_non_images_rejected_with_415(payload, name):
    with pytest.raises(ImageValidationError) as e:
        decode(payload, name, "image/jpeg")
    assert e.value.code == "unsupported_format" and e.value.http_status == 415


def test_truncated_jpeg_rejected_as_corrupt():
    data = f.jpeg(f.textured())
    with pytest.raises(ImageValidationError) as e:
        decode(data[: len(data) // 2])
    assert e.value.code == "corrupt_image" and e.value.http_status == 422


def test_garbage_with_jpeg_magic_rejected_as_corrupt():
    with pytest.raises(ImageValidationError) as e:
        decode(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    assert e.value.code == "corrupt_image"


def test_extension_mismatch_is_noted_not_rejected():
    data = f.encode(f.textured(), "PNG")
    d = decode(data, "photo.jpg", "image/jpeg")
    assert d.format == "PNG"
    assert "PNG" in d.format_mismatch


def test_generic_content_type_is_accepted():
    d = decode(f.jpeg(f.textured()), "upload", "application/octet-stream")
    assert d.format == "JPEG" and d.format_mismatch is None


def test_missing_content_type_and_filename_accepted():
    d = decode(f.jpeg(f.textured()), None, None)
    assert d.format == "JPEG"


def test_sniffing_is_independent_of_names():
    assert sniff_format(f.encode(f.textured(), "WEBP")) == "WEBP"
    assert sniff_format(b"GIF89a....") == "GIF"
    assert sniff_format(b"random") is None
