"""Tests for visualization helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ultrasound.utils.visualization import (
    create_mask_overlay,
    plot_preprocessing_comparison,
    plot_training_history,
    visualize_results,
)


def test_visualize_results_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="at least one"):
        visualize_results([], [])


def test_visualize_results_rejects_mismatched_titles(sample_float_image) -> None:
    with pytest.raises(ValueError, match="same number of items"):
        visualize_results([sample_float_image], ["original", "processed"])


def test_plot_preprocessing_comparison_requires_named_images(sample_float_image) -> None:
    with pytest.raises(ValueError, match="at least one named image"):
        plot_preprocessing_comparison(sample_float_image, {})


def test_plot_training_history_requires_metrics() -> None:
    with pytest.raises(ValueError, match="at least one metric name"):
        plot_training_history({"loss": [0.4, 0.2]}, metrics=())


def test_create_mask_overlay_validates_color_triplet(sample_float_image) -> None:
    image_rgb = np.repeat(sample_float_image[:, :, None], 3, axis=2)
    mask = np.zeros(sample_float_image.shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly three channels"):
        create_mask_overlay(image_rgb, mask, color=(1.0, 0.0))
