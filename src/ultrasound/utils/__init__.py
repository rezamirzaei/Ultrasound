"""
Utility functions for ultrasound image processing.

Includes:
- Image I/O operations
- Visualization helpers
- Metrics computation
"""

from .io import load_image, save_image, load_dicom
from .metrics import (
    compute_dice,
    compute_iou,
    compute_accuracy,
    compute_confusion_matrix,
    compute_auc_roc,
)
from .visualization import (
    visualize_results,
    plot_preprocessing_comparison,
    plot_segmentation_overlay,
    plot_training_history,
    plot_roc_curve,
)

__all__ = [
    'load_image',
    'save_image',
    'load_dicom',
    'compute_dice',
    'compute_iou',
    'compute_accuracy',
    'compute_confusion_matrix',
    'compute_auc_roc',
    'visualize_results',
    'plot_preprocessing_comparison',
    'plot_segmentation_overlay',
    'plot_training_history',
    'plot_roc_curve',
]
