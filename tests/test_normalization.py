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


def test_normalize_image_minmax_handles_constant_images() -> None:
    normalized = normalize_image(np.full((4, 4), 7.0), method="minmax", target_range=(10.0, 20.0))

    np.testing.assert_array_equal(normalized, np.full((4, 4), 10.0))


def test_normalize_image_zscore_centers_image() -> None:
    image = np.array([[0.0, 1.0], [2.0, 3.0]])

    normalized = normalize_image(image, method="zscore")

    assert np.isclose(float(np.mean(normalized)), 0.0, atol=1e-7)
    assert np.isclose(float(np.std(normalized)), 1.0, atol=1e-7)


def test_normalize_image_robust_clips_outliers() -> None:
    image = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 1000.0]])

    normalized = normalize_image(image, method="robust", target_range=(-1.0, 1.0))

    assert float(normalized.min()) >= -1.0
    assert float(normalized.max()) <= 1.0


def test_normalize_image_rejects_unknown_method(sample_float_image) -> None:
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_image(sample_float_image, method="unknown")


def test_standardize_image_accepts_scalar_channel_stats(sample_float_image) -> None:
    standardized = standardize_image(sample_float_image, mean=np.array([0.5]), std=np.array([0.25]))

    assert standardized.shape == (64, 64, 3)


def test_standardize_image_rejects_incompatible_stats(sample_rgb) -> None:
    with pytest.raises(ValueError, match="one value per image channel"):
        standardize_image(sample_rgb.astype(np.float64) / 255.0, mean=np.array([0.1, 0.2]))


def test_standardize_image_rejects_nonpositive_std(sample_float_image) -> None:
    with pytest.raises(ValueError, match="positive"):
        standardize_image(sample_float_image, std=np.array([0.0]))


def test_standardize_image_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="2-D grayscale or 3-D multi-channel"):
        standardize_image(np.array([0.1, 0.2, 0.3], dtype=np.float64))


def test_depth_compensation_handles_zero_image() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)

    compensated = depth_compensation(image)

    np.testing.assert_array_equal(compensated, image)


def test_depth_compensation_rejects_invalid_inputs(sample_rgb) -> None:
    with pytest.raises(ValueError, match="2-D grayscale"):
        depth_compensation(sample_rgb)

    with pytest.raises(ValueError, match="non-negative"):
        depth_compensation(np.zeros((8, 8), dtype=np.uint8), attenuation_coefficient=-0.1)


def test_depth_compensation_brightens_deeper_rows() -> None:
    image = np.full((16, 4), 100, dtype=np.uint8)

    compensated = depth_compensation(image, attenuation_coefficient=1.0)

    assert compensated[-1].mean() > compensated[0].mean()
