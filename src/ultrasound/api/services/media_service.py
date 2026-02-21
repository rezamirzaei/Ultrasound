"""Reusable media encoding helpers for API services."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


class MediaService:
    """Convert numpy images into frontend-consumable data URLs."""

    def as_png_data_url(self, image: np.ndarray) -> str:
        """Encode grayscale/RGB arrays as PNG data URLs."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            pil = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape for encoding: {image.shape}")

        buffer = BytesIO()
        pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
