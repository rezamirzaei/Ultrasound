"""
Ultrasound Imaging Toolkit
==========================

A comprehensive Python toolkit for ultrasound image processing, analysis, and machine learning.

This package demonstrates expertise in:
- Signal processing for ultrasound imaging
- Speckle reduction and image enhancement
- Medical image segmentation using deep learning
- Optimization-based methods (ADMM, Total Variation)

Author: Reza Mirzaeifard, PhD
Email: reza.mirzaeifard@gmail.com
"""

__version__ = "1.0.0"
__author__ = "Reza Mirzaeifard"

from .models import UltrasoundClassifier as UltrasoundClassifier
from .models import UNet as UNet
from .preprocessing import ContrastEnhancer as ContrastEnhancer
from .preprocessing import SpeckleReducer as SpeckleReducer
from .utils import load_image as load_image
from .utils import save_image as save_image
from .utils import visualize_results as visualize_results

__all__ = [
    "__author__",
    "__version__",
    "ContrastEnhancer",
    "SpeckleReducer",
    "UNet",
    "UltrasoundClassifier",
    "load_image",
    "save_image",
    "visualize_results",
]
