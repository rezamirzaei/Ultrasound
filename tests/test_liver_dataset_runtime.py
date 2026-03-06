"""Runtime-oriented tests for liver dataset download and parsing helpers."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultrasound.data.liver_dataset import (
    _download_file,
    _extract_polygon_points,
    create_synthetic_liver_dataset,
    download_liver_dataset,
    summarize_dataset,
)


class _ResponseStub:
    def __init__(self, chunks: list[bytes], *, content_length: int | None = None) -> None:
        self._chunks = list(chunks)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self) -> _ResponseStub:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def test_extract_polygon_points_accepts_point_dicts_and_tuples() -> None:
    points = _extract_polygon_points(
        {
            "shapes": [
                {"points": [{"x": 1, "y": 2}, (3, 4), [5, 6]]},
            ]
        }
    )

    assert points == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def test_summarize_dataset_without_annotations_reports_zero_boxes(tmp_path: Path) -> None:
    paths = create_synthetic_liver_dataset(tmp_path / "liver", n_samples=2)
    paths.annotations_csv.unlink()

    summary = summarize_dataset(paths)

    assert summary["images"] == 2
    assert summary["annotated_images"] == 0
    assert summary["total_boxes"] == 0
    assert summary["liver_boxes"] == 0
    assert summary["mass_boxes"] == 0


def test_download_file_writes_atomically(monkeypatch, tmp_path: Path) -> None:
    payload = b"zip-bytes"
    dest = tmp_path / "dataset.zip"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ResponseStub([payload], content_length=len(payload)),
    )

    _download_file("https://example.com/dataset.zip", dest)

    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".zip.tmp").exists()


def test_download_file_cleans_temp_file_on_failure(monkeypatch, tmp_path: Path) -> None:
    dest = tmp_path / "dataset.zip"

    def _fail(*_args: object, **_kwargs: object) -> _ResponseStub:
        raise RuntimeError("download failed")

    monkeypatch.setattr("urllib.request.urlopen", _fail)

    with pytest.raises(RuntimeError, match="download failed"):
        _download_file("https://example.com/dataset.zip", dest)

    assert not dest.exists()
    assert not dest.with_suffix(".zip.tmp").exists()


def test_download_liver_dataset_force_rebuilds_generated_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        image = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image, format="PNG")
        archive.writestr("7272660/Benign/Benign/image/case_001.png", image.getvalue())
        archive.writestr(
            "7272660/Benign/Benign/segmentation/liver/case_001.json",
            "[[1, 1], [6, 1], [6, 6], [1, 6]]",
        )

    def _fake_download(_url: str, dest: Path) -> None:
        dest.write_bytes(archive_bytes.getvalue())

    monkeypatch.setattr("ultrasound.data.liver_dataset._download_file", _fake_download)

    paths = download_liver_dataset(tmp_path / "liver_ultrasound_detection", force=True)
    stale_csv = paths.annotations_csv
    stale_flat_file = paths.train_images_dir / "stale.txt"
    stale_csv.write_text("stale\n", encoding="utf-8")
    stale_flat_file.parent.mkdir(parents=True, exist_ok=True)
    stale_flat_file.write_text("stale\n", encoding="utf-8")

    refreshed = download_liver_dataset(paths.root, force=True)

    assert refreshed.annotations_csv.read_text(encoding="utf-8").startswith("image_id,")
    assert stale_flat_file.exists() is False
    assert any(refreshed.train_images_dir.iterdir())
