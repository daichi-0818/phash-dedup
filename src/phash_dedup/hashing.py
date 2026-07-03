"""DCT-based perceptual hashing (pHash) for images.

The algorithm:
1. Load the image, convert to grayscale.
2. Resize to a fixed square size larger than the target hash size
   (default 32x32), so we have enough resolution for a DCT.
3. Compute the 2D Discrete Cosine Transform of the pixel matrix.
4. Keep the top-left ``hash_size x hash_size`` low-frequency block
   (excluding the DC term at [0, 0], which just encodes average
   brightness and adds no discriminative signal).
5. Threshold each coefficient against the median of the block to
   produce a bit: 1 if the coefficient is above the median, else 0.
6. Pack the bits into a single Python int (a 64-bit hash for the
   default 8x8 block size).

This is a standard, well known approach (popularized by pHash.org /
imagehash) reimplemented here from scratch using only Pillow and
numpy, with no external phash dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_HASH_SIZE = 8
DEFAULT_IMAGE_SIZE = 32


def _dct2(matrix: np.ndarray) -> np.ndarray:
    """2D DCT-II implemented via two sequential 1D DCTs (no scipy dependency)."""
    return _dct1(_dct1(matrix.T).T)


def _dct1(matrix: np.ndarray) -> np.ndarray:
    """1D DCT-II applied along the last axis of a 2D array.

    Implemented directly from the DCT-II definition using matrix
    multiplication against a cosine basis matrix, so the only
    dependency is numpy.
    """
    n = matrix.shape[-1]
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi / n * (i + 0.5) * k)
    return matrix @ basis.T


def image_to_phash(
    path: str | Path,
    hash_size: int = DEFAULT_HASH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> int:
    """Compute a perceptual hash (pHash) for the image at ``path``.

    Args:
        path: path to an image file readable by Pillow.
        hash_size: side length of the retained low-frequency DCT block.
            The resulting hash has ``hash_size * hash_size`` bits.
        image_size: side length the image is resized to before the DCT.
            Must be >= hash_size.

    Returns:
        An unsigned integer with ``hash_size * hash_size`` bits encoding
        the perceptual hash.
    """
    if image_size < hash_size:
        raise ValueError("image_size must be >= hash_size")

    with Image.open(path) as img:
        gray = img.convert("L").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.float64)

    return _matrix_to_phash(pixels, hash_size=hash_size)


def _matrix_to_phash(pixels: np.ndarray, hash_size: int = DEFAULT_HASH_SIZE) -> int:
    dct = _dct2(pixels)

    # Keep the low-frequency block, drop the DC term (top-left coefficient).
    low_freq = dct[:hash_size, :hash_size]
    coeffs = low_freq.flatten()[1:]
    median = np.median(coeffs)

    bits = (low_freq.flatten() > median).astype(np.uint8)
    # Force the DC bit to 0 so it never contributes noise.
    bits[0] = 0

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def bits_for_hash_size(hash_size: int = DEFAULT_HASH_SIZE) -> int:
    """Number of bits produced for a given ``hash_size``."""
    return hash_size * hash_size
