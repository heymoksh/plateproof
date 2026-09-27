"""Cryptographic and perceptual image hashes, implemented with NumPy only.

- SHA-256 identifies byte-identical files.
- pHash (DCT-based) and dHash (gradient-based) are 64-bit perceptual hashes:
  visually similar images (resized, recompressed) have a small Hamming distance.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from PIL import Image

HASH_BITS = 64


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0, :] = np.sqrt(1 / n)
    return m


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def phash(img: Image.Image) -> int:
    pixels = _gray(img, (32, 32))
    c = _dct_matrix(32)
    low = (c @ pixels @ c.T)[:8, :8]
    median = np.median(low.flatten()[1:])  # exclude the DC term
    return _bits_to_int(low > median)


def dhash(img: Image.Image) -> int:
    pixels = _gray(img, (9, 8))
    return _bits_to_int(pixels[:, 1:] > pixels[:, :-1])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def to_hex(value: int) -> str:
    return f"{value:016x}"


def from_hex(value: str) -> int:
    return int(value, 16)


def detail_level(img: Image.Image) -> float:
    """Standard deviation of a small grayscale copy. Near-uniform images
    (blank walls, solid colours) make perceptual hashes unreliable."""
    return float(_gray(img, (64, 64)).std())
