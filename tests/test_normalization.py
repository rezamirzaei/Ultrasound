"""Tests for normalization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ultrasound.preprocessing.normalization import (
    depth_compensation,
    normalize_image,
    standardize_image,
)


def test_normalize_image_rejects_descending_target_range(sample_float_image) -> None:
    with pytest.raises(ValueError, match="ascending order"):
        normalize_image(sample_float_image, method="minmax", target_range=(1.0, 0.0))


def test_standardize_image_accepts_scalar_channel_stats(sample_float_image) -> None:
    standardized = standardize_image(sample_float_image, mean=np.array([0.5]), std=np.array([0.25]))

    assert standardized.shape == (64, 64, 3)


def test_standardize_image_rejects_incompatible_stats(sample_rgb) -> None:
    with pytest.raises(ValueError, match="one value per image channel"):
        standardize_image(sample_rgb.astype(np.float64) / 255.0, mean=np.array([0.1, 0.2]))


def test_depth_compensation_handles_zero_image() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)

    compensated = depth_compensation(image)

    np.testing.assert_array_equal(compensated, image)


def test_depth_compensation_rejects_invalid_inputs(sample_rgb) -> None:
    with pytest.raises(ValueError, match="2-D grayscale"):
        depth_compensation(sample_rgb)

    with pytest.raises(ValueError, match="non-negative"):
        depth_compensation(np.zeros((8, 8), dtype=np.uint8), attenuation_coefficient=-0.1)
