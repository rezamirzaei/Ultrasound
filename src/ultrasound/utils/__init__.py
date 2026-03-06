"""
Utility functions for ultrasound image processing.

Includes:
- Image I/O operations
- Visualization helpers
- Metrics computation
"""

from .io import load_dicom, load_image, load_nifti, save_image
from .metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_confusion_matrix,
    compute_dice,
    compute_iou,
)
from .visualization import (
    plot_preprocessing_comparison,
    plot_roc_curve,
    plot_segmentation_overlay,
    plot_training_history,
    visualize_results,
)

__all__ = [
    "load_image",
    "save_image",
    "load_dicom",
    "load_nifti",
    "compute_dice",
    "compute_iou",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_auc_roc",
    "visualize_results",
    "plot_preprocessing_comparison",
    "plot_segmentation_overlay",
    "plot_training_history",
    "plot_roc_curve",
]
