"""Tests for preprocessing.speckle module."""

import numpy as np
import pytest

from ultrasound.preprocessing.speckle import (
    SpeckleReducer,
    adaptive_median_filter,
    estimate_noise_variance,
    frost_filter,
    lee_filter,
    wiener_filter,
)


class TestSpeckleReducer:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            SpeckleReducer(method="invalid")

    def test_even_window_size(self):
        with pytest.raises(ValueError):
            SpeckleReducer(window_size=4)

    def test_lee_output_shape(self, sample_grayscale):
        reducer = SpeckleReducer(method="lee", window_size=5)
        assert reducer.reduce(sample_grayscale).shape == sample_grayscale.shape

    def test_frost_output_shape(self, sample_grayscale):
        reducer = SpeckleReducer(method="frost", window_size=5)
        assert reducer.reduce(sample_grayscale).shape == sample_grayscale.shape

    def test_median_output_shape(self, sample_grayscale):
        reducer = SpeckleReducer(method="median", window_size=5)
        assert reducer.reduce(sample_grayscale).shape == sample_grayscale.shape

    def test_rgb_input(self, sample_rgb):
        reducer = SpeckleReducer(method="lee", window_size=5)
        result = reducer.reduce(sample_rgb)
        assert result.ndim == 2
        assert result.shape == sample_rgb.shape[:2]

    def test_estimate_speckle_level(self, sample_grayscale):
        reducer = SpeckleReducer()
        mean_val, cv = reducer.estimate_speckle_level(sample_grayscale)
        assert mean_val > 0
        assert cv >= 0

    def test_estimate_speckle_level_handles_zero_mean(self):
        reducer = SpeckleReducer()
        mean_val, cv = reducer.estimate_speckle_level(np.zeros((8, 8), dtype=np.uint8))
        assert mean_val == 0
        assert cv == 0

    def test_wiener_and_adaptive_median_methods(self, sample_grayscale):
        wiener_result = SpeckleReducer(method="wiener", window_size=5).reduce(sample_grayscale)
        adaptive_result = SpeckleReducer(method="adaptive_median", window_size=5).reduce(
            sample_grayscale
        )
        assert wiener_result.shape == sample_grayscale.shape
        assert adaptive_result.shape == sample_grayscale.shape


class TestLeeFilter:
    def test_constant_image(self):
        image = np.full((32, 32), 100, dtype=np.float64)
        result = lee_filter(image, window_size=5)
        np.testing.assert_array_almost_equal(result, image, decimal=0)

    def test_explicit_noise_variance_is_supported(self):
        image = np.full((16, 16), 100, dtype=np.float64)
        result = lee_filter(image, window_size=3, noise_variance=0.05)
        assert result.shape == image.shape


class TestFrostFilter:
    def test_output_shape(self, sample_grayscale):
        result = frost_filter(sample_grayscale.astype(np.float64), window_size=5)
        assert result.shape == sample_grayscale.shape


class TestAdaptiveMedianFilter:
    def test_output_shape(self, sample_grayscale):
        result = adaptive_median_filter(sample_grayscale.astype(np.float64))
        assert result.shape == sample_grayscale.shape

    def test_unresolved_pixels_fall_back_to_max_window_median(self):
        image = np.full((9, 9), 100.0)
        result = adaptive_median_filter(image, min_window_size=3, max_window_size=5)
        np.testing.assert_array_equal(result, image)


def test_wiener_filter_accepts_explicit_noise_variance(sample_grayscale):
    result = wiener_filter(sample_grayscale.astype(np.float64), window_size=5, noise_variance=2.0)
    assert result.shape == sample_grayscale.shape


def test_estimate_noise_variance_returns_nonnegative_value(sample_grayscale):
    variance = estimate_noise_variance(sample_grayscale.astype(np.float64))
    assert variance >= 0.0
