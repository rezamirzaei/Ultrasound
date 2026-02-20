"""Tests for utils.metrics module."""

import numpy as np
import pytest

from ultrasound.utils.metrics import (
    compute_accuracy,
    compute_dice,
    compute_iou,
    compute_mae,
    compute_psnr,
    compute_rmse,
    compute_segmentation_metrics,
    compute_sensitivity,
    compute_specificity,
    compute_ssim,
)


class TestDice:
    def test_identical_masks(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        assert compute_dice(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_masks(self):
        pred = np.zeros((32, 32), dtype=np.uint8)
        target = np.ones((32, 32), dtype=np.uint8)
        assert compute_dice(pred, target) < 0.01

    def test_partial_overlap(self, binary_mask_pair):
        pred, target = binary_mask_pair
        assert 0 < compute_dice(pred, target) < 1


class TestIoU:
    def test_identical_masks(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        assert compute_iou(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_dice_geq_iou(self, binary_mask_pair):
        pred, target = binary_mask_pair
        assert compute_dice(pred, target) >= compute_iou(pred, target)


class TestSensitivitySpecificity:
    def test_perfect_prediction(self):
        mask = np.ones((16, 16), dtype=np.uint8)
        assert compute_sensitivity(mask, mask) == pytest.approx(1.0, abs=1e-5)
        assert compute_specificity(np.zeros_like(mask), np.zeros_like(mask)) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_accuracy(self):
        pred = np.array([0, 1, 1, 0])
        target = np.array([0, 1, 0, 0])
        assert compute_accuracy(pred, target) == pytest.approx(0.75)


class TestRegressionMetrics:
    def test_rmse_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(arr, arr) == pytest.approx(0.0)

    def test_mae_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert compute_mae(arr, arr) == pytest.approx(0.0)

    def test_psnr_identical(self):
        arr = np.array([100.0, 200.0])
        assert compute_psnr(arr, arr) == float("inf")

    def test_ssim_identical(self):
        arr = np.random.default_rng(42).random((32, 32))
        assert compute_ssim(arr, arr) == pytest.approx(1.0, abs=1e-3)


class TestSegmentationMetrics:
    def test_returns_all_keys(self, binary_mask_pair):
        pred, target = binary_mask_pair
        metrics = compute_segmentation_metrics(pred, target)
        for key in (
            "dice",
            "iou",
            "hausdorff_95",
            "pixel_accuracy",
            "sensitivity",
            "specificity",
        ):
            assert key in metrics
