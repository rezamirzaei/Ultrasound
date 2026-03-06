"""Tests for image I/O helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
from PIL import Image

from ultrasound.utils.io import load_dicom, load_image, load_nifti, save_image


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


def test_load_image_supports_grayscale_resize(tmp_path: Path) -> None:
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    path = tmp_path / "gray.png"
    Image.fromarray(image).save(path)

    loaded = load_image(path, grayscale=True, target_size=(2, 3))

    assert loaded.shape == (2, 3)
    assert loaded.dtype == np.uint8


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


def test_save_image_round_trips_rgb_arrays(tmp_path: Path) -> None:
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "rgb.png"

    save_image(image, path, normalize=False)
    loaded = load_image(path)

    np.testing.assert_array_equal(loaded, image)


def test_load_dicom_applies_rescale_and_window(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sample.dcm"
    path.write_bytes(b"placeholder")
    fake_module = ModuleType("pydicom")

    class _FakeDataset:
        pixel_array = np.array([[0, 50], [100, 150]], dtype=np.float64)
        RescaleSlope = 2.0
        RescaleIntercept = -10.0

    def _fake_dcmread(_path: str) -> _FakeDataset:
        return _FakeDataset()

    fake_module.dcmread = _fake_dcmread  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydicom", fake_module)

    result = load_dicom(path, window_center=60.0, window_width=120.0)

    assert result.shape == (2, 2)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_load_nifti_extracts_requested_slice(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sample.nii.gz"
    path.write_bytes(b"placeholder")
    fake_module = ModuleType("nibabel")

    class _FakeImage:
        def get_fdata(self) -> np.ndarray:
            volume = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
            return volume

    def _fake_load(_path: str) -> _FakeImage:
        return _FakeImage()

    fake_module.load = _fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nibabel", fake_module)

    result = load_nifti(path, slice_idx=1)

    assert result.shape == (3, 4)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert 250 <= int(result.max()) <= 255
