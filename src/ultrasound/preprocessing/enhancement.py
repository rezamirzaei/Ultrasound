"""
Contrast Enhancement for Ultrasound Images.

Ultrasound images often suffer from low contrast due to:
- Limited dynamic range of tissue reflectivity
- Depth-dependent attenuation
- Speckle noise affecting visibility

This module implements various contrast enhancement techniques:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Standard histogram equalization
- Gamma correction
- Logarithmic transformation

References:
    Pizer, S.M., et al. (1987). Adaptive histogram equalization and its variations.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ContrastEnhancer:
    """
    Comprehensive contrast enhancement processor for ultrasound images.
    
    Provides a unified interface for various contrast enhancement methods
    with automatic parameter selection based on image characteristics.
    
    Example:
        >>> enhancer = ContrastEnhancer(method='clahe')
        >>> enhanced = enhancer.enhance(ultrasound_image)
    """
    
    AVAILABLE_METHODS = ['clahe', 'histogram_eq', 'gamma', 'logarithmic', 'adaptive']
    
    def __init__(
        self,
        method: str = 'clahe',
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.0,
    ):
        """
        Initialize contrast enhancer.
        
        Args:
            method: Enhancement method ('clahe', 'histogram_eq', 'gamma', 'logarithmic', 'adaptive')
            clip_limit: Contrast limiting threshold for CLAHE
            tile_grid_size: Grid size for CLAHE
            gamma: Gamma value for gamma correction (< 1 brightens, > 1 darkens)
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {self.AVAILABLE_METHODS}")
        
        self.method = method
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply contrast enhancement to an ultrasound image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Contrast-enhanced image
        """
        if self.method == 'clahe':
            return apply_clahe(image, self.clip_limit, self.tile_grid_size)
        elif self.method == 'histogram_eq':
            return histogram_equalization(image)
        elif self.method == 'gamma':
            return gamma_correction(image, self.gamma)
        elif self.method == 'logarithmic':
            return logarithmic_transform(image)
        elif self.method == 'adaptive':
            return adaptive_enhancement(image)
            
    def analyze_contrast(self, image: np.ndarray) -> dict:
        """
        Analyze contrast characteristics of an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with contrast metrics
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        return {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
            'dynamic_range': float(np.max(gray) - np.min(gray)),
            'contrast_ratio': float(np.std(gray) / (np.mean(gray) + 1e-10)),
            'histogram_entropy': float(compute_histogram_entropy(gray)),
        }


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE divides the image into small tiles and performs histogram
    equalization on each tile independently. The contrast limiting
    prevents over-amplification of noise in homogeneous regions.
    
    This is particularly effective for ultrasound images because:
    - Handles depth-dependent intensity variations
    - Limits noise amplification in uniform regions
    - Preserves local contrast in regions of interest
    
    Args:
        image: Input image
        clip_limit: Contrast limiting threshold (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image
    """
    # Handle RGB images
    if image.ndim == 3:
        # Convert to LAB color space
        if image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Process first channel only
            gray = image[:, :, 0]
    else:
        gray = image
    
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply standard histogram equalization.
    
    Redistributes pixel intensities to utilize the full dynamic range,
    resulting in a more uniform histogram distribution.
    
    Args:
        image: Input grayscale or RGB image
        
    Returns:
        Histogram-equalized image
    """
    if image.ndim == 3:
        if image.shape[2] == 3:
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
        
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
    return cv2.equalizeHist(gray)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction for contrast adjustment.
    
    Gamma correction applies a power-law transformation:
        output = input^gamma
    
    - gamma < 1: Brightens the image (useful for dark ultrasound images)
    - gamma > 1: Darkens the image (increases contrast in bright regions)
    
    Args:
        image: Input image
        gamma: Gamma value (default 1.0 = no change)
        
    Returns:
        Gamma-corrected image
    """
    # Normalize to [0, 1]
    normalized = image.astype(np.float64) / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to [0, 255]
    return (corrected * 255).astype(np.uint8)


def logarithmic_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Apply logarithmic transformation for contrast enhancement.
    
    The log transform expands dark pixels and compresses bright pixels:
        output = c * log(1 + input)
    
    This is particularly useful for ultrasound images with a wide
    dynamic range, as it can reveal details in darker regions.
    
    Args:
        image: Input image
        c: Scaling constant
        
    Returns:
        Log-transformed image
    """
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Apply log transform
    log_transformed = c * np.log1p(img_float)
    
    # Normalize to [0, 255]
    log_transformed = (log_transformed / log_transformed.max()) * 255
    
    return log_transformed.astype(np.uint8)


def adaptive_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Adaptive contrast enhancement based on image characteristics.
    
    Automatically selects and applies the most appropriate enhancement
    based on analysis of the input image.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image
    
    # Analyze image characteristics
    mean_intensity = np.mean(gray)
    contrast_ratio = np.std(gray) / (mean_intensity + 1e-10)
    
    if mean_intensity < 80:
        # Dark image: use gamma correction to brighten
        gamma = 0.7
        enhanced = gamma_correction(gray, gamma)
    elif contrast_ratio < 0.3:
        # Low contrast: use CLAHE
        enhanced = apply_clahe(gray, clip_limit=3.0)
    else:
        # Normal: mild CLAHE
        enhanced = apply_clahe(gray, clip_limit=2.0)
    
    return enhanced


def compute_histogram_entropy(image: np.ndarray) -> float:
    """
    Compute the entropy of the image histogram.
    
    Higher entropy indicates more uniform distribution of intensities.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Histogram entropy value
    """
    # Compute histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    
    # Normalize to probability distribution
    hist = hist / hist.sum()
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy
