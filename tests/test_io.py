"""Tests for image I/O helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ultrasound.utils.io import load_image, save_image


def test_load_image_reads_rgb_data(tmp_path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [255, 128, 64]
    path = tmp_path / "sample.png"
    Image.fromarray(image).save(path)

    loaded = load_image(path)

    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)


def test_load_image_rejects_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load image"):
        load_image(path)


def test_load_image_validates_target_size(tmp_path: Path) -> None:
    image = np.zeros((4, 5), dtype=np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(image).save(path)

    with pytest.raises(ValueError, match="positive height and width"):
        load_image(path, grayscale=True, target_size=(0, 8))


def test_save_image_clips_out_of_range_values(tmp_path: Path) -> None:
    image = np.array([[300.0, -10.0], [128.0, 42.0]], dtype=np.float32)
    path = tmp_path / "saved.png"

    save_image(image, path, normalize=True)
    loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, np.array([[255, 0], [128, 42]], dtype=np.uint8))


def test_save_image_raises_when_backend_write_fails(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "blocked.png"

    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(OSError, match="Failed to save image"):
        save_image(np.zeros((2, 2), dtype=np.uint8), path)
