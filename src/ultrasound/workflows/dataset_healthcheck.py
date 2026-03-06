"""Reusable dataset healthcheck workflow for notebook and script consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetHealthcheckResult:
    busi_counts: dict[str, int]
    overlays: dict[str, np.ndarray]
    ndt_rows: list[dict[str, Any]]
    report: dict[str, Any]
    health_status: dict[str, bool]


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = np.asarray(image, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if mask_arr.ndim != 2:
        raise ValueError("mask must have shape [H, W]")

    overlay = image_rgb.astype(np.float32) / 255.0
    mask_bool = mask_arr > 0
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] *= 0.3
    overlay[mask_bool, 2] *= 0.3
    return np.asarray(np.clip(overlay * 255.0, 0.0, 255.0), dtype=np.uint8)


def run_dataset_healthcheck(
    busi_counts: dict[str, int],
    sample_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    ndt_rows: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> DatasetHealthcheckResult:
    overlays = {
        class_name: _mask_overlay(image, mask)
        for class_name, (image, mask) in sample_pairs.items()
    }

    health_status = {
        "busi_ready": int(busi_counts.get("benign", 0)) > 0
        and int(busi_counts.get("malignant", 0)) > 0,
        "ndt_ready": len(ndt_rows) > 0,
    }
    health_status["overall_ready"] = bool(
        health_status["busi_ready"] and health_status["ndt_ready"]
    )

    report = {
        "seed": int(seed),
        "busi_counts": {key: int(value) for key, value in busi_counts.items()},
        "busi_total": int(sum(int(value) for value in busi_counts.values())),
        "ndt_samples": ndt_rows,
    }
    return DatasetHealthcheckResult(
        busi_counts={key: int(value) for key, value in busi_counts.items()},
        overlays=overlays,
        ndt_rows=list(ndt_rows),
        report=report,
        health_status=health_status,
    )
