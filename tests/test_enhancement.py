"""Tests for preprocessing.enhancement module."""

from __future__ import annotations

import numpy as np
import pytest

import ultrasound.preprocessing.enhancement as enhancement
from ultrasound.preprocessing.enhancement import (
    ContrastEnhancer,
    adaptive_enhancement,
    apply_clahe,
    compute_histogram_entropy,
    gamma_correction,
    histogram_equalization,
    logarithmic_transform,
)


class TestContrastEnhancer:
    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            ContrastEnhancer(method="bad")

    def test_clahe_output_shape(self, sample_grayscale) -> None:
        enhancer = ContrastEnhancer(method="clahe")
        assert enhancer.enhance(sample_grayscale).shape == sample_grayscale.shape

    def test_histogram_eq_method(self, sample_grayscale) -> None:
        enhancer = ContrastEnhancer(method="histogram_eq")
        result = enhancer.enhance(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_analyze_contrast(self, sample_grayscale) -> None:
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

    @pytest.mark.parametrize("method", ["gamma", "logarithmic", "adaptive"])
    def test_other_methods_return_image(self, method: str, sample_grayscale) -> None:
        enhancer = ContrastEnhancer(method=method)
        result = enhancer.enhance(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_analyze_contrast_supports_rgb(self, sample_rgb) -> None:
        enhancer = ContrastEnhancer()
        stats = enhancer.analyze_contrast(sample_rgb)
        assert stats["dynamic_range"] >= 0


class TestGammaCorrection:
    def test_identity(self, sample_grayscale) -> None:
        result = gamma_correction(sample_grayscale, gamma=1.0)
        np.testing.assert_array_almost_equal(result, sample_grayscale, decimal=0)

    def test_brightens_dark_values_for_gamma_below_one(self) -> None:
        image = np.array([[16, 64], [128, 255]], dtype=np.uint8)
        result = gamma_correction(image, gamma=0.5)
        assert result[0, 0] > image[0, 0]


class TestHistogramEntropy:
    def test_uniform_image(self) -> None:
        image = np.full((32, 32), 128, dtype=np.uint8)
        entropy = compute_histogram_entropy(image)
        assert entropy >= 0

    def test_varied_image_has_higher_entropy_than_uniform(self) -> None:
        uniform = np.full((32, 32), 128, dtype=np.uint8)
        varied = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        assert compute_histogram_entropy(varied) > compute_histogram_entropy(uniform)


def test_apply_clahe_supports_rgb(sample_rgb) -> None:
    result = apply_clahe(sample_rgb, clip_limit=2.5)
    assert result.shape == sample_rgb.shape
    assert result.dtype == np.uint8


def test_histogram_equalization_supports_rgb(sample_rgb) -> None:
    result = histogram_equalization(sample_rgb)
    assert result.shape == sample_rgb.shape
    assert result.dtype == np.uint8


def test_apply_clahe_uses_first_channel_for_non_rgb_images() -> None:
    image = np.dstack(
        [
            np.full((8, 8), 10, dtype=np.uint8),
            np.full((8, 8), 20, dtype=np.uint8),
        ]
    )
    result = apply_clahe(image)
    assert result.shape == image.shape[:2]


def test_histogram_equalization_uses_first_channel_for_non_rgb_images() -> None:
    image = np.dstack(
        [
            np.tile(np.arange(8, dtype=np.uint8), (8, 1)),
            np.zeros((8, 8), dtype=np.uint8),
        ]
    )
    result = histogram_equalization(image)
    assert result.shape == image.shape[:2]


def test_logarithmic_transform_normalizes_output() -> None:
    image = np.array([[0, 1], [16, 255]], dtype=np.uint8)
    result = logarithmic_transform(image)
    assert result.dtype == np.uint8
    assert result[0, 1] > image[0, 1]
    assert result.max() == 255


def test_adaptive_enhancement_uses_gamma_for_dark_images(monkeypatch) -> None:
    image = np.full((16, 16), 10, dtype=np.uint8)
    captured: dict[str, float] = {}

    def _fake_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
        captured["gamma"] = gamma
        return np.zeros_like(gray)

    monkeypatch.setattr(enhancement, "gamma_correction", _fake_gamma)

    result = adaptive_enhancement(image)

    assert captured["gamma"] == pytest.approx(0.7)
    np.testing.assert_array_equal(result, np.zeros_like(image))


def test_adaptive_enhancement_uses_stronger_clahe_for_low_contrast(monkeypatch) -> None:
    image = np.full((16, 16), 180, dtype=np.uint8)
    captured: dict[str, float] = {}

    def _fake_clahe(
        gray: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        captured["clip_limit"] = clip_limit
        return np.zeros_like(gray)

    monkeypatch.setattr(enhancement, "apply_clahe", _fake_clahe)

    adaptive_enhancement(image)

    assert captured["clip_limit"] == pytest.approx(3.0)


def test_adaptive_enhancement_uses_mild_clahe_for_normal_images(monkeypatch) -> None:
    image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    captured: dict[str, float] = {}

    def _fake_clahe(
        gray: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        captured["clip_limit"] = clip_limit
        return np.zeros_like(gray)

    monkeypatch.setattr(enhancement, "apply_clahe", _fake_clahe)

    adaptive_enhancement(image)

    assert captured["clip_limit"] == pytest.approx(2.0)
