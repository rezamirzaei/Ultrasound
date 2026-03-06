"""Tests for reusable notebook-oriented workflows."""

from __future__ import annotations

import numpy as np

from ultrasound.workflows import run_model_metric_smoke, run_preprocessing_workbench


def _sample_image() -> np.ndarray:
    x = np.linspace(0, 255, 64, dtype=np.uint8)
    grid_x, grid_y = np.meshgrid(x, x)
    image = np.stack([grid_x, grid_y, ((grid_x.astype(np.uint16) + grid_y.astype(np.uint16)) // 2).astype(np.uint8)], axis=2)
    return image.astype(np.uint8)


def _sample_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:46, 20:44] = 255
    return mask


def test_run_preprocessing_workbench_returns_expected_artifacts() -> None:
    result = run_preprocessing_workbench(_sample_image(), n_iter=8)

    assert result.gray.shape == (64, 64)
    assert set(result.processed) == {"Lee", "Frost", "Median", "CLAHE", "ADMM-TV"}
    assert set(result.quality) == {"lee", "frost", "tv"}
    assert set(result.speckle_cv) == {"original", "lee", "frost", "tv"}
    assert result.convergence["primal_residuals"]
    for image in result.processed.values():
        assert image.shape == result.gray.shape


def test_run_model_metric_smoke_returns_numeric_summaries() -> None:
    result = run_model_metric_smoke(_sample_image(), _sample_mask(), seed=7)

    assert result.shapes["unet_logits"] == [2, 1, 128, 128]
    assert result.shapes["cnn_logits"][0] == 2
    assert set(result.losses) == {"dice_loss", "combined_loss", "focal_loss"}
    assert result.segmentation_metrics["iou"] >= 0.0
    assert result.confusion_matrix.shape == (2, 2)
    assert "accuracy" in result.classification_metrics
