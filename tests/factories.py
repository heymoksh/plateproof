"""Builders for realistic test images (textured content, real EXIF structures)."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import piexif
from PIL import Image, ImageDraw, PngImagePlugin


def textured(width: int = 640, height: int = 480, seed: int = 0) -> Image.Image:
    """A photo-like image: gradient, random shapes and sensor-like noise.

    Different seeds give visually different images, so perceptual hashes are
    meaningfully distinct (flat test images would collide).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            x / width * rng.uniform(80, 200),
            y / height * rng.uniform(80, 200),
            (x + y) / (width + height) * rng.uniform(80, 200),
        ],
        axis=-1,
    )
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x1, y1 = x0 + int(rng.integers(20, width // 2)), y0 + int(rng.integers(20, height // 2))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=colour)
    arr = np.asarray(img, dtype=np.float64) + rng.normal(0, 4, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _ts(value: datetime | str | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.strftime("%Y:%m:%d %H:%M:%S")
    return value.encode()


def camera_exif(
    make: str = "Apple",
    model: str = "iPhone 15",
    taken: datetime | str | None = "2026:05:01 10:00:00",
    modified: datetime | str | None = None,
    software: str | None = None,
    gps: bool = True,
    thumbnail: Image.Image | None = None,
    orientation: int | None = None,
) -> bytes:
    zeroth = {piexif.ImageIFD.Make: make.encode(), piexif.ImageIFD.Model: model.encode()}
    if software:
        zeroth[piexif.ImageIFD.Software] = software.encode()
    if orientation:
        zeroth[piexif.ImageIFD.Orientation] = orientation
    modified_ts = _ts(modified) or _ts(taken)
    if modified_ts:
        zeroth[piexif.ImageIFD.DateTime] = modified_ts
    exif_ifd = {}
    if taken is not None:
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = _ts(taken)
        exif_ifd[piexif.ExifIFD.DateTimeDigitized] = _ts(taken)
    gps_ifd = {}
    if gps:
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: ((12, 1), (41, 1), (3000, 100)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSLongitude: ((79, 1), (58, 1), (1200, 100)),
        }
    first, thumb_bytes = {}, None
    if thumbnail is not None:
        buf = io.BytesIO()
        thumbnail.save(buf, "JPEG", quality=85)
        thumb_bytes = buf.getvalue()
        first = {piexif.ImageIFD.Compression: 6}
    return piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps_ifd, "1st": first, "thumbnail": thumb_bytes})


def thumbnail_of(img: Image.Image, size=(160, 120)) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail(size)
    return thumb


def jpeg(img: Image.Image, exif: bytes | None = None, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    kwargs = {"quality": quality}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, "JPEG", **kwargs)
    return buf.getvalue()


def png(img: Image.Image, text: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    info = None
    if text:
        info = PngImagePlugin.PngInfo()
        for key, value in text.items():
            info.add_text(key, value)
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def camera_photo(seed: int = 1, **exif_kwargs) -> bytes:
    """A camera-original style JPEG: textured, full EXIF, GPS and a matching thumbnail."""
    img = textured(seed=seed)
    exif_kwargs.setdefault("thumbnail", thumbnail_of(img))
    return jpeg(img, camera_exif(**exif_kwargs))
