from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos((np.pi / size) * (x + 0.5) * k)
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)


def image_metadata_and_phash(path: str | Path) -> tuple[int, int, str, str]:
    """Return width, height, format, and a 64-bit DCT perceptual hash."""
    with Image.open(path) as source:
        image_format = (source.format or "unknown").lower()
        image = ImageOps.exif_transpose(source).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        width, height = source.size
        pixels = np.asarray(image, dtype=np.float64)
    dct = _dct_matrix(32)
    low = (dct @ pixels @ dct.T)[:8, :8]
    median = np.median(low.reshape(-1)[1:])
    bits = (low >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return width, height, image_format, f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()

