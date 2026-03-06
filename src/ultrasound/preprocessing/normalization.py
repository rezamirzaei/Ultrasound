"""
Image Normalization and Standardization for Ultrasound Images.

Proper normalization is crucial for:
- Consistent input to machine learning models
- Compensation of equipment variations
- Handling different acquisition settings
"""

from __future__ import annotations

from typing import cast

import numpy as np

TargetRange = tuple[float, float]


def _validate_target_range(target_range: TargetRange) -> TargetRange:
    target_min, target_max = target_range
    if target_min >= target_max:
        raise ValueError("target_range must be in ascending order")
    return float(target_min), float(target_max)


def _prepare_channel_stats(values: np.ndarray | None, default: tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(default if values is None else values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array")
    if name == "std" and np.any(array <= 0):
        raise ValueError("std values must be positive")
    return array


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    target_range: TargetRange = (0, 1),
) -> np.ndarray:
    """
    Normalize image intensities to a target range.

    Args:
        image: Input image
        method: Normalization method ('minmax', 'zscore', 'robust')
        target_range: Target range for minmax normalization

    Returns:
        Normalized image
    """
    img = image.astype(np.float64)
    target_min, target_max = _validate_target_range(target_range)

    if method == "minmax":
        # Min-max normalization
        min_val = img.min()
        max_val = img.max()

        if max_val - min_val > 0:
            normalized = (img - min_val) / (max_val - min_val)
            normalized = normalized * (target_max - target_min) + target_min
        else:
            normalized = np.full_like(img, target_min)

    elif method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(img)
        std = np.std(img)
        normalized = (img - mean) / (std + 1e-10)

    elif method == "robust":
        # Robust normalization using percentiles
        p5 = np.percentile(img, 5)
        p95 = np.percentile(img, 95)

        if p95 - p5 > 0:
            normalized = np.clip((img - p5) / (p95 - p5), 0, 1)
            normalized = normalized * (target_max - target_min) + target_min
        else:
            normalized = np.full_like(img, target_min)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return cast(np.ndarray, normalized)


def standardize_image(
    image: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Standardize image using ImageNet statistics or provided values.

    This is commonly used when using pretrained networks (ResNet, VGG, etc.)
    that were trained on ImageNet.

    Args:
        image: Input image (expected to be in [0, 1] range)
        mean: Mean values for each channel (default: ImageNet means)
        std: Std values for each channel (default: ImageNet stds)

    Returns:
        Standardized image
    """
    mean_values = _prepare_channel_stats(mean, (0.485, 0.456, 0.406), "mean")
    std_values = _prepare_channel_stats(std, (0.229, 0.224, 0.225), "std")

    img = image.astype(np.float64)

    # Ensure image is in [0, 1] range
    if img.max() > 1.0:
        img = img / 255.0

    # Handle grayscale images
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim != 3:
        raise ValueError("image must be 2-D grayscale or 3-D multi-channel")

    channel_count = img.shape[2]
    if mean_values.size not in (1, channel_count):
        raise ValueError("mean must contain either one value or one value per image channel")
    if std_values.size not in (1, channel_count):
        raise ValueError("std must contain either one value or one value per image channel")

    if mean_values.size == 1:
        mean_values = np.repeat(mean_values, channel_count)
    if std_values.size == 1:
        std_values = np.repeat(std_values, channel_count)

    # Standardize
    standardized = (img - mean_values.reshape(1, 1, -1)) / std_values.reshape(1, 1, -1)

    return cast(np.ndarray, standardized)


def depth_compensation(
    image: np.ndarray,
    attenuation_coefficient: float = 0.5,
) -> np.ndarray:
    """
    Compensate for depth-dependent attenuation in ultrasound images.

    Ultrasound intensity decreases with depth due to:
    - Absorption by tissue
    - Scattering
    - Beam spreading

    This function applies Time Gain Compensation (TGC) to correct for
    depth-dependent signal loss.

    Args:
        image: Input ultrasound image (2D grayscale)
        attenuation_coefficient: Attenuation coefficient (dB/cm/MHz)

    Returns:
        Depth-compensated image
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    if attenuation_coefficient < 0:
        raise ValueError("attenuation_coefficient must be non-negative")

    img = image.astype(np.float64)

    # Create depth-dependent gain
    depth = np.arange(img.shape[0])
    gain = np.exp(attenuation_coefficient * depth / img.shape[0])

    # Apply gain to each row
    compensated = img * gain[:, np.newaxis]

    # Normalize to original range
    peak = float(compensated.max())
    if peak <= 0.0:
        return np.zeros_like(img, dtype=np.uint8)
    compensated = np.clip((compensated / peak) * 255.0, 0.0, 255.0)

    return cast(np.ndarray, compensated.astype(np.uint8))
