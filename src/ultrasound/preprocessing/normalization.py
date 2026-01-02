"""
Image Normalization and Standardization for Ultrasound Images.

Proper normalization is crucial for:
- Consistent input to machine learning models
- Compensation of equipment variations
- Handling different acquisition settings
"""

import numpy as np
from typing import Tuple, Optional


def normalize_image(
    image: np.ndarray,
    method: str = 'minmax',
    target_range: Tuple[float, float] = (0, 1),
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
    
    if method == 'minmax':
        # Min-max normalization
        min_val = img.min()
        max_val = img.max()
        
        if max_val - min_val > 0:
            normalized = (img - min_val) / (max_val - min_val)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = np.full_like(img, target_range[0])
            
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(img)
        std = np.std(img)
        normalized = (img - mean) / (std + 1e-10)
        
    elif method == 'robust':
        # Robust normalization using percentiles
        p5 = np.percentile(img, 5)
        p95 = np.percentile(img, 95)
        
        if p95 - p5 > 0:
            normalized = np.clip((img - p5) / (p95 - p5), 0, 1)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = np.full_like(img, target_range[0])
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized


def standardize_image(
    image: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
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
    # ImageNet statistics
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    
    img = image.astype(np.float64)
    
    # Ensure image is in [0, 1] range
    if img.max() > 1.0:
        img = img / 255.0
    
    # Handle grayscale images
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # Standardize
    standardized = (img - mean) / std
    
    return standardized


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
    img = image.astype(np.float64)
    
    # Create depth-dependent gain
    depth = np.arange(img.shape[0])
    gain = np.exp(attenuation_coefficient * depth / img.shape[0])
    
    # Apply gain to each row
    compensated = img * gain[:, np.newaxis]
    
    # Normalize to original range
    compensated = (compensated / compensated.max()) * 255
    
    return compensated.astype(np.uint8)
