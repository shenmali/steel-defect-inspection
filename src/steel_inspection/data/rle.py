"""Run-length encoding helpers for Severstal masks."""

import numpy as np


def decode_rle(encoded: str | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode a one-indexed, Fortran-order run-length encoded mask."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not encoded or encoded != encoded:
        return mask.reshape(shape, order="F")

    starts_and_lengths = np.asarray(encoded.split(), dtype=np.int64).reshape(-1, 2)
    for start, length in starts_and_lengths:
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(shape, order="F")
