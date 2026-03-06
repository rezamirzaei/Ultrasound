"""Tests for visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from ultrasound.utils.visualization import (
    create_comparison_overlay,
    create_mask_overlay,
    plot_confusion_matrix,
    plot_preprocessing_comparison,
    plot_roc_curve,
    plot_segmentation_overlay,
    plot_speckle_analysis,
    plot_training_history,
    visualize_results,
)


def test_visualize_results_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="at least one"):
        visualize_results([], [])


def test_visualize_results_rejects_mismatched_titles(sample_float_image) -> None:
    with pytest.raises(ValueError, match="same number of items"):
        visualize_results([sample_float_image], ["original", "processed"])


def test_visualize_results_saves_figure(tmp_path: Path, sample_float_image, sample_rgb) -> None:
    save_path = tmp_path / "results.png"

    fig = visualize_results([sample_float_image, sample_rgb], ["gray", "rgb"], save_path=save_path)

    assert save_path.exists()
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_preprocessing_comparison_requires_named_images(sample_float_image) -> None:
    with pytest.raises(ValueError, match="at least one named image"):
        plot_preprocessing_comparison(sample_float_image, {})


def test_plot_segmentation_overlay_supports_prediction_and_saves(
    tmp_path: Path,
    sample_float_image,
) -> None:
    save_path = tmp_path / "segmentation.png"
    mask_true = np.zeros(sample_float_image.shape, dtype=np.uint8)
    mask_pred = np.zeros(sample_float_image.shape, dtype=np.uint8)
    mask_true[8:24, 8:24] = 1
    mask_pred[16:32, 16:32] = 1

    fig = plot_segmentation_overlay(
        sample_float_image,
        mask_true,
        mask_pred=mask_pred,
        save_path=save_path,
    )

    assert save_path.exists()
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_training_history_requires_metrics() -> None:
    with pytest.raises(ValueError, match="at least one metric name"):
        plot_training_history({"loss": [0.4, 0.2]}, metrics=())


def test_plot_training_history_saves_requested_metrics(tmp_path: Path) -> None:
    save_path = tmp_path / "history.png"

    fig = plot_training_history(
        {
            "loss": [0.6, 0.3],
            "val_loss": [0.7, 0.4],
            "accuracy": [0.5, 0.8],
            "val_accuracy": [0.45, 0.75],
        },
        metrics=("loss", "accuracy"),
        save_path=save_path,
    )

    assert save_path.exists()
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_roc_curve_handles_binary_scores(tmp_path: Path) -> None:
    save_path = tmp_path / "roc_binary.png"

    fig = plot_roc_curve(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.4, 0.35, 0.9]),
        save_path=save_path,
    )

    assert save_path.exists()
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_plot_roc_curve_handles_multiclass_scores(tmp_path: Path) -> None:
    save_path = tmp_path / "roc_multi.png"

    fig = plot_roc_curve(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array(
            [
                [0.95, 0.03, 0.02],
                [0.02, 0.95, 0.03],
                [0.02, 0.05, 0.93],
                [0.8, 0.1, 0.1],
                [0.1, 0.75, 0.15],
                [0.05, 0.2, 0.75],
            ]
        ),
        class_names=("A", "B", "C"),
        save_path=save_path,
    )

    assert save_path.exists()
    assert len(fig.axes[0].lines) == 4
    plt.close(fig)


def test_plot_confusion_matrix_can_render_raw_counts(tmp_path: Path) -> None:
    save_path = tmp_path / "cm.png"

    fig = plot_confusion_matrix(
        np.array([[5, 1], [2, 4]]),
        class_names=("neg", "pos"),
        normalize=False,
        save_path=save_path,
    )

    assert save_path.exists()
    assert any(text.get_text() == "5" for text in fig.axes[0].texts)
    plt.close(fig)


def test_plot_speckle_analysis_saves_report(tmp_path: Path, sample_rgb) -> None:
    save_path = tmp_path / "speckle.png"

    fig = plot_speckle_analysis(sample_rgb, window_size=8, save_path=save_path)

    assert save_path.exists()
    assert any(axis.get_title() == "Speckle Analysis" for axis in fig.axes)
    plt.close(fig)


def test_create_mask_overlay_validates_color_triplet(sample_float_image) -> None:
    image_rgb = np.repeat(sample_float_image[:, :, None], 3, axis=2)
    mask = np.zeros(sample_float_image.shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly three channels"):
        create_mask_overlay(image_rgb, mask, color=(1.0, 0.0))


def test_create_comparison_overlay_marks_tp_fp_and_fn() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float64)
    mask_true = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    mask_pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    overlay = create_comparison_overlay(image, mask_true, mask_pred, alpha=1.0)

    np.testing.assert_array_equal(overlay[0, 0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(overlay[0, 1], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(overlay[1, 0], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(overlay[1, 1], np.array([0.0, 0.0, 0.0]))
