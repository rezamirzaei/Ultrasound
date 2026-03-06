"""
Image I/O utilities for ultrasound images.

Supports common formats and DICOM medical imaging format.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np

ImagePath = str | Path
ImageSize = tuple[int, int]


def load_image(
    path: ImagePath,
    grayscale: bool = False,
    target_size: ImageSize | None = None,
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
    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), read_flag)

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("target_size must contain positive height and width")
        img = cv2.resize(img, (target_size[1], target_size[0]))

    return cast(np.ndarray, img)


def save_image(
    image: np.ndarray,
    path: ImagePath,
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
        img = np.rint(np.clip(img, 0.0, 1.0) * 255.0)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(path), img):
        raise OSError(f"Failed to save image: {path}")


def load_dicom(
    path: ImagePath,
    window_center: float | None = None,
    window_width: float | None = None,
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
            "pydicom is required to load DICOM files. " "Install it with: pip install pydicom"
        )

    path = Path(path)
    ds = pydicom.dcmread(str(path))

    # Get pixel array
    img = ds.pixel_array.astype(np.float64)

    # Apply rescale if available
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
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

    return cast(np.ndarray, img.astype(np.uint8))


def load_nifti(
    path: ImagePath,
    slice_idx: int | None = None,
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
            "nibabel is required to load NIfTI files. " "Install it with: pip install nibabel"
        )

    path = Path(path)
    nii = nib.load(str(path))
    img = nii.get_fdata()

    if slice_idx is not None and img.ndim == 3:
        img = img[:, :, slice_idx]

    # Normalize to [0, 255]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255

    return cast(np.ndarray, img.astype(np.uint8))
