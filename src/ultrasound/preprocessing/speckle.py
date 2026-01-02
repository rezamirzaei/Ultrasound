"""
Speckle Noise Reduction for Ultrasound Images.

Speckle noise is an inherent characteristic of ultrasound imaging, arising from
the constructive and destructive interference of backscattered echoes from
tissue microstructure. This module implements various speckle reduction
techniques to improve image quality while preserving important diagnostic features.

Theory:
-------
Speckle follows a multiplicative noise model:
    I(x,y) = R(x,y) * n(x,y)

where I is the observed image, R is the true reflectivity, and n is the
multiplicative speckle noise.

Implemented Methods:
- Lee Filter: Minimum Mean Square Error (MMSE) estimator
- Frost Filter: Exponentially damped convolutional filter
- Median Filter: Non-linear smoothing preserving edges
- Wiener Filter: Adaptive noise reduction

References:
    Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics.
    Frost, V.S., et al. (1982). A model for radar images and its application to adaptive digital filtering.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import wiener
from typing import Optional, Tuple
import cv2


class SpeckleReducer:
    """
    Comprehensive speckle reduction processor for ultrasound images.
    
    This class provides a unified interface for various speckle reduction
    methods, with automatic parameter tuning based on estimated noise levels.
    
    Example:
        >>> reducer = SpeckleReducer(method='lee', window_size=7)
        >>> denoised = reducer.reduce(ultrasound_image)
    """
    
    AVAILABLE_METHODS = ['lee', 'frost', 'median', 'wiener', 'adaptive_median']
    
    def __init__(
        self,
        method: str = 'lee',
        window_size: int = 7,
        damping_factor: float = 1.0,
        noise_variance: Optional[float] = None,
    ):
        """
        Initialize speckle reducer.
        
        Args:
            method: Speckle reduction method ('lee', 'frost', 'median', 'wiener', 'adaptive_median')
            window_size: Size of the local window for filtering (must be odd)
            damping_factor: Damping factor for Frost filter (higher = more smoothing)
            noise_variance: Known noise variance (if None, estimated from image)
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {self.AVAILABLE_METHODS}")
        
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        self.method = method
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.noise_variance = noise_variance
        
    def reduce(self, image: np.ndarray) -> np.ndarray:
        """
        Apply speckle reduction to an ultrasound image.
        
        Args:
            image: Input grayscale image (2D array) or RGB image (3D array)
            
        Returns:
            Filtered image with reduced speckle noise
        """
        # Handle RGB images by processing each channel or converting to grayscale
        if image.ndim == 3:
            # Process as grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
            filtered = self._apply_filter(gray.astype(np.float64))
            return np.clip(filtered, 0, 255).astype(np.uint8)
        
        return np.clip(self._apply_filter(image.astype(np.float64)), 0, 255).astype(np.uint8)
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply the selected filter method."""
        if self.method == 'lee':
            return lee_filter(image, self.window_size, self.noise_variance)
        elif self.method == 'frost':
            return frost_filter(image, self.window_size, self.damping_factor)
        elif self.method == 'median':
            return median_speckle_filter(image, self.window_size)
        elif self.method == 'wiener':
            return wiener_filter(image, self.window_size, self.noise_variance)
        elif self.method == 'adaptive_median':
            return adaptive_median_filter(image, max_window_size=self.window_size)
        
    def estimate_speckle_level(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Estimate speckle noise level in an ultrasound image.
        
        Uses the Equivalent Number of Looks (ENL) metric commonly used
        in radar and ultrasound speckle analysis.
        
        Args:
            image: Input ultrasound image
            
        Returns:
            Tuple of (mean, coefficient_of_variation)
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        mean_val = np.mean(image)
        std_val = np.std(image)
        cv = std_val / mean_val if mean_val > 0 else 0
        
        return mean_val, cv


def lee_filter(
    image: np.ndarray,
    window_size: int = 7,
    noise_variance: Optional[float] = None,
) -> np.ndarray:
    """
    Lee speckle filter based on local statistics.
    
    The Lee filter is a minimum mean square error (MMSE) estimator that
    adaptively smooths the image based on local variance estimates.
    
    The filter equation is:
        R̂ = μ + W * (I - μ)
    
    where:
        W = var(R) / (var(R) + var(n))
        
    For areas with low variance (homogeneous regions), W → 0 and output → mean
    For areas with high variance (edges), W → 1 and output → original
    
    Args:
        image: Input grayscale image
        window_size: Size of local window for statistics calculation
        noise_variance: Known noise variance (if None, estimated globally)
        
    Returns:
        Filtered image
    """
    img = image.astype(np.float64)
    
    # Calculate local mean using uniform filter
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(img, kernel)
    
    # Calculate local variance
    local_sq_mean = ndimage.convolve(img ** 2, kernel)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Ensure non-negative
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Use coefficient of variation method
        # For fully developed speckle, CV ≈ 1
        global_mean = np.mean(img)
        global_var = np.var(img)
        noise_variance = global_var / (global_mean ** 2) if global_mean > 0 else 0.1
    
    # Calculate weight
    noise_var_scaled = noise_variance * (local_mean ** 2)
    
    # Avoid division by zero
    denominator = local_var + noise_var_scaled + 1e-10
    weight = local_var / denominator
    weight = np.clip(weight, 0, 1)
    
    # Apply Lee filter
    output = local_mean + weight * (img - local_mean)
    
    return output


def frost_filter(
    image: np.ndarray,
    window_size: int = 7,
    damping_factor: float = 1.0,
) -> np.ndarray:
    """
    Frost speckle filter with exponential damping.
    
    The Frost filter uses an exponentially damped convolution kernel where
    the damping is controlled by local coefficient of variation. Areas with
    high CV (edges) receive less smoothing.
    
    Kernel: K(t) = A * exp(-damping * Ci * |t|)
    
    where Ci is the local coefficient of variation.
    
    Args:
        image: Input grayscale image
        window_size: Size of local window
        damping_factor: Controls the rate of exponential decay (higher = more edge preservation)
        
    Returns:
        Filtered image
    """
    img = image.astype(np.float64)
    half_size = window_size // 2
    
    # Pad image
    padded = np.pad(img, half_size, mode='reflect')
    output = np.zeros_like(img)
    
    # Calculate local statistics
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(img, kernel)
    local_sq_mean = ndimage.convolve(img ** 2, kernel)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    
    # Coefficient of variation
    local_cv = np.sqrt(local_var) / (local_mean + 1e-10)
    
    # Create distance matrix for kernel
    y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
    dist = np.sqrt(x**2 + y**2)
    
    # Apply filter pixel by pixel (vectorized version)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ci = local_cv[i, j]
            
            # Frost kernel with exponential damping
            frost_kernel = np.exp(-damping_factor * ci * dist)
            frost_kernel = frost_kernel / np.sum(frost_kernel)
            
            # Extract local patch and apply kernel
            patch = padded[i:i+window_size, j:j+window_size]
            output[i, j] = np.sum(patch * frost_kernel)
    
    return output


def median_speckle_filter(
    image: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Median filter for speckle reduction.
    
    The median filter is a non-linear filter that replaces each pixel with
    the median of neighboring pixels. It's effective at removing impulse
    noise while preserving edges.
    
    Args:
        image: Input grayscale image
        window_size: Size of local window (must be odd)
        
    Returns:
        Filtered image
    """
    return ndimage.median_filter(image.astype(np.float64), size=window_size)


def wiener_filter(
    image: np.ndarray,
    window_size: int = 5,
    noise_variance: Optional[float] = None,
) -> np.ndarray:
    """
    Wiener filter for speckle reduction.
    
    The Wiener filter is an optimal linear filter that minimizes the mean
    squared error between the estimated and true signal.
    
    Args:
        image: Input grayscale image
        window_size: Size of local window
        noise_variance: Known noise variance (if None, estimated from image)
        
    Returns:
        Filtered image
    """
    img = image.astype(np.float64)
    
    if noise_variance is None:
        # Estimate noise from high-frequency components
        noise_variance = estimate_noise_variance(img)
    
    return wiener(img, mysize=window_size, noise=noise_variance)


def adaptive_median_filter(
    image: np.ndarray,
    min_window_size: int = 3,
    max_window_size: int = 7,
) -> np.ndarray:
    """
    Adaptive median filter with variable window size.
    
    This filter adapts its window size based on local noise levels.
    It starts with a small window and increases it until a non-impulse
    median value is found.
    
    Args:
        image: Input grayscale image
        min_window_size: Minimum window size
        max_window_size: Maximum window size
        
    Returns:
        Filtered image
    """
    img = image.astype(np.float64)
    output = np.copy(img)
    
    pad_size = max_window_size // 2
    padded = np.pad(img, pad_size, mode='reflect')
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window_size = min_window_size
            
            while window_size <= max_window_size:
                half = window_size // 2
                
                # Extract window centered at current pixel
                pi = i + pad_size
                pj = j + pad_size
                window = padded[pi-half:pi+half+1, pj-half:pj+half+1]
                
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = img[i, j]
                
                # Check if median is an impulse
                if z_min < z_med < z_max:
                    # Median is not an impulse
                    if z_min < z_xy < z_max:
                        # Current pixel is not an impulse
                        output[i, j] = z_xy
                    else:
                        # Current pixel is an impulse, replace with median
                        output[i, j] = z_med
                    break
                else:
                    # Increase window size
                    window_size += 2
            
            # If max window reached, use median
            if window_size > max_window_size:
                half = max_window_size // 2
                pi = i + pad_size
                pj = j + pad_size
                window = padded[pi-half:pi+half+1, pj-half:pj+half+1]
                output[i, j] = np.median(window)
    
    return output


def estimate_noise_variance(image: np.ndarray) -> float:
    """
    Estimate noise variance using the Median Absolute Deviation (MAD) method.
    
    Uses a Laplacian filter to isolate high-frequency noise components.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Estimated noise variance
    """
    # Laplacian kernel for high-frequency detection
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Apply Laplacian
    edges = ndimage.convolve(image.astype(np.float64), laplacian)
    
    # MAD estimator (robust to outliers)
    mad = np.median(np.abs(edges - np.median(edges)))
    
    # Convert MAD to standard deviation (assuming Gaussian)
    sigma = mad / 0.6745
    
    return sigma ** 2
