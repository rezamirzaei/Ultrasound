"""Feature extraction utilities for lightweight industrial image classifiers."""

from __future__ import annotations

import numpy as np
from PIL import Image


class IndustrialFeatureExtractor:
    """Build fixed-size handcrafted features from industrial RGB images."""

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = np.asarray(image_rgb, dtype=np.float32).mean(axis=2)
        gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)

        small = Image.fromarray(gray, mode="L").resize((48, 48), Image.Resampling.BILINEAR)
        small_arr = np.asarray(small, dtype=np.float32) / 255.0

        grad_x = np.diff(small_arr, axis=1, append=small_arr[:, -1:])
        grad_y = np.diff(small_arr, axis=0, append=small_arr[-1:, :])
        grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))

        percentile_features = np.percentile(small_arr, [10, 25, 50, 75, 90]).astype(np.float32)
        stats = np.array(
            [
                float(np.mean(small_arr)),
                float(np.std(small_arr)),
                float(np.mean(grad_mag)),
                float(np.std(grad_mag)),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [small_arr.reshape(-1), grad_mag.reshape(-1), percentile_features, stats],
            axis=0,
        )
