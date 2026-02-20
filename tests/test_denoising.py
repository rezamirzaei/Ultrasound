"""Tests for preprocessing.denoising module."""

import numpy as np
from scipy.sparse import issparse

from ultrasound.preprocessing.denoising import (
    _build_difference_operators,
    admm_tv_denoising,
    bilateral_filter,
    soft_threshold,
    total_variation_denoising,
)


class TestSoftThreshold:
    def test_zero_threshold_is_identity(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(soft_threshold(x, 0.0), x)

    def test_large_threshold_zeros_all(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(soft_threshold(x, 10.0), np.zeros(5))

    def test_partial_threshold(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected = np.array([-1.5, 0.0, 0.0, 0.0, 1.5])
        np.testing.assert_array_almost_equal(soft_threshold(x, 1.5), expected)


class TestBuildDifferenceOperators:
    def test_output_shapes(self):
        dx, dy = _build_difference_operators(8, 10)
        assert dx.shape == (80, 80)
        assert dy.shape == (80, 80)

    def test_sparse(self):
        dx, dy = _build_difference_operators(4, 4)
        assert issparse(dx) and issparse(dy)

    def test_constant_image_zero_gradient(self):
        m, n = 4, 5
        dx, dy = _build_difference_operators(m, n)
        u = np.ones(m * n)
        np.testing.assert_array_almost_equal(dx @ u, 0)
        np.testing.assert_array_almost_equal(dy @ u, 0)


class TestTotalVariationDenoising:
    def test_output_shape_and_dtype(self, sample_grayscale):
        result = total_variation_denoising(sample_grayscale, n_iter=5)
        assert result.shape == sample_grayscale.shape
        assert result.dtype == np.uint8

    def test_output_bounds(self, sample_grayscale):
        result = total_variation_denoising(sample_grayscale, n_iter=5)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_unchanged(self):
        image = np.full((32, 32), 128, dtype=np.uint8)
        np.testing.assert_array_almost_equal(
            total_variation_denoising(image, n_iter=20),
            image,
            decimal=0,
        )


class TestAdmmTvDenoising:
    def test_output_shape_dtype(self, sample_grayscale):
        result, _ = admm_tv_denoising(sample_grayscale, n_iter=5)
        assert result.shape == sample_grayscale.shape
        assert result.dtype == np.uint8

    def test_convergence_info_keys(self, sample_grayscale):
        _, info = admm_tv_denoising(sample_grayscale, n_iter=10)
        for key in ("primal_residuals", "dual_residuals", "n_iter", "rho_history"):
            assert key in info

    def test_convergence_arrays_are_finite(self, sample_grayscale):
        _, info = admm_tv_denoising(sample_grayscale, n_iter=20)
        assert len(info["primal_residuals"]) > 0
        assert len(info["dual_residuals"]) > 0
        assert np.isfinite(np.asarray(info["primal_residuals"])).all()
        assert np.isfinite(np.asarray(info["dual_residuals"])).all()

    def test_early_stopping(self, sample_grayscale):
        _, info = admm_tv_denoising(sample_grayscale, n_iter=200, abstol=1e-2, reltol=1e-1)
        assert info["n_iter"] < 200

    def test_adaptive_rho_changes_history(self, sample_grayscale):
        _, info = admm_tv_denoising(sample_grayscale, n_iter=30, adaptive_rho=True)
        assert len(info["rho_history"]) >= 1


class TestBilateralFilter:
    def test_output_shape(self, sample_grayscale):
        result = bilateral_filter(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_constant_image_unchanged(self):
        image = np.full((32, 32), 128, dtype=np.uint8)
        np.testing.assert_array_equal(bilateral_filter(image), image)
