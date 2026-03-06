"""
Preprocessing module for ultrasound images.

This module implements various preprocessing techniques commonly used in
ultrasound image processing:
- Speckle reduction (Lee filter, Frost filter, Median filter)
- Contrast enhancement (CLAHE, histogram equalization)
- Noise reduction using Total Variation denoising
- Image normalization and standardization

These techniques are essential for improving ultrasound image quality
before applying machine learning algorithms.
"""

from .denoising import admm_tv_denoising, bilateral_filter, total_variation_denoising
from .enhancement import ContrastEnhancer, apply_clahe, gamma_correction, histogram_equalization
from .normalization import depth_compensation, normalize_image, standardize_image
from .speckle import SpeckleReducer, frost_filter, lee_filter, median_speckle_filter

__all__ = [
    "SpeckleReducer",
    "lee_filter",
    "frost_filter",
    "median_speckle_filter",
    "ContrastEnhancer",
    "apply_clahe",
    "histogram_equalization",
    "gamma_correction",
    "depth_compensation",
    "normalize_image",
    "standardize_image",
    "total_variation_denoising",
    "bilateral_filter",
    "admm_tv_denoising",
]
