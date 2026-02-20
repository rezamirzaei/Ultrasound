"""Tests for preprocessing.enhancement module."""

import numpy as np
import pytest

from ultrasound.preprocessing.enhancement import (
    ContrastEnhancer,
    compute_histogram_entropy,
    gamma_correction,
)


class TestContrastEnhancer:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ContrastEnhancer(method="bad")

    def test_clahe_output_shape(self, sample_grayscale):
        enhancer = ContrastEnhancer(method="clahe")
        assert enhancer.enhance(sample_grayscale).shape == sample_grayscale.shape

    def test_analyze_contrast(self, sample_grayscale):
        enhancer = ContrastEnhancer()
        stats = enhancer.analyze_contrast(sample_grayscale)
        for key in (
            "mean",
            "std",
            "min",
            "max",
            "dynamic_range",
            "contrast_ratio",
            "histogram_entropy",
        ):
            assert key in stats


class TestGammaCorrection:
    def test_identity(self, sample_grayscale):
        result = gamma_correction(sample_grayscale, gamma=1.0)
        np.testing.assert_array_almost_equal(result, sample_grayscale, decimal=0)


class TestHistogramEntropy:
    def test_uniform_image(self):
        image = np.full((32, 32), 128, dtype=np.uint8)
        entropy = compute_histogram_entropy(image)
        assert entropy >= 0
