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

__version__ = "0.1.0"
__author__ = "Reza Mirzaeifard"

from .preprocessing import SpeckleReducer, ContrastEnhancer
from .models import UNet, UltrasoundClassifier
from .utils import load_image, save_image, visualize_results
