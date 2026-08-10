import numpy as np

from steel_inspection.data.rle import decode_rle


def test_decode_rle_returns_expected_fortran_mask():
    mask = decode_rle("1 2 5 1", (2, 3))

    assert mask.tolist() == [[1, 0, 1], [1, 0, 0]]


def test_decode_rle_returns_empty_mask_for_missing_encoding():
    mask = decode_rle(None, (2, 3))

    assert np.array_equal(mask, np.zeros((2, 3), dtype=np.uint8))
