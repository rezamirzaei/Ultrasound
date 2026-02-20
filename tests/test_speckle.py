"""Tests for preprocessing.speckle module."""

import numpy as np
import pytest

from ultrasound.preprocessing.speckle import (
    SpeckleReducer,
    adaptive_median_filter,
    frost_filter,
    lee_filter,
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


class TestLeeFilter:
    def test_constant_image(self):
        image = np.full((32, 32), 100, dtype=np.float64)
        result = lee_filter(image, window_size=5)
        np.testing.assert_array_almost_equal(result, image, decimal=0)


class TestFrostFilter:
    def test_output_shape(self, sample_grayscale):
        result = frost_filter(sample_grayscale.astype(np.float64), window_size=5)
        assert result.shape == sample_grayscale.shape


class TestAdaptiveMedianFilter:
    def test_output_shape(self, sample_grayscale):
        result = adaptive_median_filter(sample_grayscale.astype(np.float64))
        assert result.shape == sample_grayscale.shape
