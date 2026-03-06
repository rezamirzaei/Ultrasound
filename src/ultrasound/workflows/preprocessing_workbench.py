"""Reusable preprocessing comparison workflow used by notebooks and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ultrasound.preprocessing.denoising import admm_tv_denoising
from ultrasound.preprocessing.enhancement import ContrastEnhancer
from ultrasound.preprocessing.speckle import SpeckleReducer
from ultrasound.utils.metrics import compute_psnr, compute_rmse, compute_ssim


@dataclass(frozen=True)
class PreprocessingWorkbenchResult:
    gray: np.ndarray
    processed: dict[str, np.ndarray]
    quality: dict[str, dict[str, float]]
    speckle_cv: dict[str, float]
    mean_intensity: float
    convergence: dict[str, Any]


def run_preprocessing_workbench(
    image_rgb: np.ndarray,
    *,
    clip_limit: float = 2.5,
    lambda_tv: float = 0.06,
    rho: float = 1.0,
    n_iter: int = 35,
) -> PreprocessingWorkbenchResult:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape [H, W, 3]")

    gray = np.mean(image, axis=2).astype(np.uint8)

    lee = SpeckleReducer(method="lee", window_size=7)
    frost = SpeckleReducer(method="frost", window_size=5, damping_factor=1.5)
    median = SpeckleReducer(method="median", window_size=5)
    enhancer = ContrastEnhancer(method="clahe", clip_limit=clip_limit)

    img_lee = lee.reduce(gray)
    img_frost = frost.reduce(gray)
    img_median = median.reduce(gray)
    img_clahe = enhancer.enhance(gray)
    img_tv, convergence = admm_tv_denoising(gray, lambda_tv=lambda_tv, rho=rho, n_iter=n_iter, verbose=False)

    processed = {
        "Lee": img_lee,
        "Frost": img_frost,
        "Median": img_median,
        "CLAHE": img_clahe,
        "ADMM-TV": img_tv,
    }

    _, cv_before = lee.estimate_speckle_level(gray)
    mean_val, cv_after_lee = lee.estimate_speckle_level(img_lee)
    _, cv_after_frost = lee.estimate_speckle_level(img_frost)
    _, cv_after_tv = lee.estimate_speckle_level(img_tv)

    quality = {
        "lee": {
            "rmse": float(compute_rmse(img_lee, gray)),
            "psnr": float(compute_psnr(img_lee, gray, data_range=255.0)),
            "ssim": float(compute_ssim(img_lee, gray, data_range=255.0)),
        },
        "frost": {
            "rmse": float(compute_rmse(img_frost, gray)),
            "psnr": float(compute_psnr(img_frost, gray, data_range=255.0)),
            "ssim": float(compute_ssim(img_frost, gray, data_range=255.0)),
        },
        "tv": {
            "rmse": float(compute_rmse(img_tv, gray)),
            "psnr": float(compute_psnr(img_tv, gray, data_range=255.0)),
            "ssim": float(compute_ssim(img_tv, gray, data_range=255.0)),
        },
    }

    return PreprocessingWorkbenchResult(
        gray=gray,
        processed=processed,
        quality=quality,
        speckle_cv={
            "original": float(cv_before),
            "lee": float(cv_after_lee),
            "frost": float(cv_after_frost),
            "tv": float(cv_after_tv),
        },
        mean_intensity=float(mean_val),
        convergence=convergence,
    )
