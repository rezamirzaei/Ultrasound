"""
Image I/O utilities for ultrasound images.

Supports common formats and DICOM medical imaging format.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from PIL import Image
import cv2


def load_image(
    path: Union[str, Path],
    grayscale: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        grayscale: Load as grayscale
        target_size: Resize to (height, width) if specified
        
    Returns:
        Image as numpy array (H, W, C) or (H, W) for grayscale
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load image
    if grayscale:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Resize if needed
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    return img


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    normalize: bool = True,
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image array
        path: Output path
        normalize: Normalize to [0, 255] if needed
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img = image.copy()
    
    # Normalize if needed
    if normalize and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    elif normalize and img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(path), img)


def load_dicom(
    path: Union[str, Path],
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> np.ndarray:
    """
    Load a DICOM image file.
    
    DICOM is the standard format for medical imaging including ultrasound.
    This function handles HU windowing for proper visualization.
    
    Args:
        path: Path to DICOM file
        window_center: Window center for intensity windowing
        window_width: Window width for intensity windowing
        
    Returns:
        Image as numpy array
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required to load DICOM files. "
            "Install it with: pip install pydicom"
        )
    
    path = Path(path)
    ds = pydicom.dcmread(str(path))
    
    # Get pixel array
    img = ds.pixel_array.astype(np.float64)
    
    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    
    # Apply windowing
    if window_center is not None and window_width is not None:
        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        # Auto-scale
        img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
    
    return img.astype(np.uint8)


def load_nifti(
    path: Union[str, Path],
    slice_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Load a NIfTI image file (common for 3D medical imaging).
    
    Args:
        path: Path to NIfTI file
        slice_idx: Specific slice to load (for 3D volumes)
        
    Returns:
        Image as numpy array
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required to load NIfTI files. "
            "Install it with: pip install nibabel"
        )
    
    path = Path(path)
    nii = nib.load(str(path))
    img = nii.get_fdata()
    
    if slice_idx is not None and img.ndim == 3:
        img = img[:, :, slice_idx]
    
    # Normalize to [0, 255]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
    
    return img.astype(np.uint8)
