"""Upload validation and image normalisation.

Everything downstream works on the `DecodedImage` produced here, so each
image is decoded exactly once, with explicit limits and clear error messages.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from pathlib import PurePath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # Optional: HEIC/HEIF support (default iPhone format)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:  # pragma: no cover - depends on platform wheels
    HEIF_SUPPORTED = False

# Formats accepted for analysis, keyed by the name Pillow reports.
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "HEIF", "AVIF", "MPO"}

_CONTENT_TYPES = {
    "JPEG": {"image/jpeg", "image/jpg", "image/pjpeg"},
    "MPO": {"image/jpeg", "image/jpg"},
    "PNG": {"image/png", "image/apng"},
    "WEBP": {"image/webp"},
    "GIF": {"image/gif"},
    "BMP": {"image/bmp", "image/x-ms-bmp"},
    "TIFF": {"image/tiff"},
    "HEIF": {"image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"},
    "AVIF": {"image/avif"},
}
_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg", ".jpe", ".jfif"},
    "MPO": {".jpg", ".jpeg", ".mpo"},
    "PNG": {".png", ".apng"},
    "WEBP": {".webp"},
    "GIF": {".gif"},
    "BMP": {".bmp"},
    "TIFF": {".tif", ".tiff"},
    "HEIF": {".heic", ".heif", ".hif"},
    "AVIF": {".avif"},
}
_GENERIC_CONTENT_TYPES = {None, "", "application/octet-stream", "binary/octet-stream"}


class ImageValidationError(Exception):
    """Raised when an upload cannot be analysed. Carries a user-facing message."""

    def __init__(self, code: str, message: str, http_status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class DecodedImage:
    data: bytes
    format: str
    image: Image.Image          # first frame, as stored in the file (not rotated)
    rgb: Image.Image            # orientation-corrected, 8-bit RGB copy used for analysis
    width: int
    height: int
    mode: str
    frames: int
    orientation: int | None
    has_alpha: bool
    declared_content_type: str | None
    file_extension: str | None
    format_mismatch: str | None = None
    notes: list[str] = field(default_factory=list)


def sniff_format(data: bytes) -> str | None:
    """Identify the container from magic bytes, independent of filename or headers."""
    head = data[:32]
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if head[:2] == b"BM":
        return "BMP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "AVIF"
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"):
            return "HEIF"
    return None


def _describe_mismatch(fmt: str, content_type: str | None, extension: str | None) -> str | None:
    problems = []
    if content_type not in _GENERIC_CONTENT_TYPES and content_type not in _CONTENT_TYPES.get(fmt, set()):
        problems.append(f"declared content type '{content_type}'")
    if extension and extension not in _EXTENSIONS.get(fmt, set()):
        problems.append(f"file extension '{extension}'")
    if not problems:
        return None
    return f"File content is {fmt}, but the {' and '.join(problems)} suggest otherwise."


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to 8-bit RGB without distorting content.

    Transparent areas are composited onto white rather than silently becoming
    black, and high-bit-depth images are scaled rather than clipped.
    """
    mode = img.mode
    if mode in ("P", "PA"):
        img = img.convert("RGBA")
        mode = "RGBA"
    if mode in ("RGBA", "LA", "La", "RGBa"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if mode.startswith("I") or mode == "F":
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if mode.startswith("I;16"):
            arr = arr / 65535.0 * 255.0
        elif hi > 255 or lo < 0 or mode == "F":
            arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").convert("RGB")
    if mode != "RGB":
        return img.convert("RGB")
    return img


def validate_and_decode(
    data: bytes,
    filename: str | None,
    content_type: str | None,
    max_bytes: int,
    max_pixels: int,
) -> DecodedImage:
    if not data:
        raise ImageValidationError("empty_file", "The uploaded file is empty.", 400)
    if len(data) > max_bytes:
        raise ImageValidationError(
            "file_too_large",
            f"The file is {len(data) / 1_048_576:.1f} MB; the limit is {max_bytes / 1_048_576:.0f} MB.",
            413,
        )

    sniffed = sniff_format(data)
    if sniffed is None:
        raise ImageValidationError(
            "unsupported_format",
            "This file is not a supported image. Upload a JPEG, PNG, WebP, HEIC, AVIF, TIFF, BMP or GIF file.",
            415,
        )
    if sniffed == "HEIF" and not HEIF_SUPPORTED:
        raise ImageValidationError(
            "heic_not_supported",
            "HEIC images need the 'pillow-heif' package. Install it with: pip install pillow-heif",
            415,
        )

    extension = PurePath(filename).suffix.lower() if filename else None
    content_type = (content_type or "").split(";")[0].strip().lower() or None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            width, height = img.size
            if width * height > max_pixels:
                raise ImageValidationError(
                    "too_many_pixels",
                    f"The image is {width}x{height} ({width * height / 1e6:.0f} MP); "
                    f"the limit is {max_pixels / 1e6:.0f} MP.",
                    413,
                )
            img.load()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("too_many_pixels", "The image dimensions are too large to analyse safely.", 413)
    except UnidentifiedImageError:
        raise ImageValidationError(
            "corrupt_image", f"The file looks like {sniffed} but could not be decoded. It may be corrupted.", 422
        )
    except (OSError, SyntaxError, ValueError, EOFError) as exc:
        detail = "It appears to be truncated or corrupted." if "truncated" in str(exc).lower() else "It may be corrupted."
        raise ImageValidationError("corrupt_image", f"The {sniffed} image could not be fully decoded. {detail}", 422)

    fmt = img.format or sniffed
    if fmt not in SUPPORTED_FORMATS:
        raise ImageValidationError("unsupported_format", f"{fmt} images are not supported.", 415)

    notes: list[str] = []
    frames = getattr(img, "n_frames", 1) or 1
    if frames > 1:
        notes.append(f"Animated or multi-frame file ({frames} frames); only the first frame was analysed.")

    orientation = None
    try:
        orientation = img.getexif().get(0x0112)
    except Exception:
        pass

    try:
        oriented = ImageOps.exif_transpose(img)
    except Exception:
        oriented = img
        notes.append("EXIF orientation could not be applied; the image was analysed as stored.")
    if orientation and orientation != 1:
        notes.append(f"Applied EXIF orientation {orientation} before analysis.")

    has_alpha = img.mode in ("RGBA", "LA", "La", "RGBa", "PA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        notes.append("Transparent areas were placed on a white background for analysis.")

    mismatch = _describe_mismatch(fmt if fmt != "MPO" else "JPEG", content_type, extension)
    if mismatch:
        notes.append(mismatch)

    return DecodedImage(
        data=data,
        format=fmt,
        image=img,
        rgb=to_rgb(oriented),
        width=width,
        height=height,
        mode=img.mode,
        frames=frames,
        orientation=orientation,
        has_alpha=has_alpha,
        declared_content_type=content_type,
        file_extension=extension,
        format_mismatch=mismatch,
        notes=notes,
    )
