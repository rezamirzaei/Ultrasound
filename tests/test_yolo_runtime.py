"""Runtime-oriented tests for YOLO helpers and trainer backends."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ultrasound.api.models.schemas import DownloadedAssetManifest
from ultrasound.api.services.yolo_trainer import (
    YoloDatasetPreparer,
    YoloTrainer,
    YoloTrainingConfig,
)
from ultrasound.api.services.yolo_utils import (
    download_url_to_path,
    load_manifest,
    mask_to_xyxy,
    parse_yolo_txt_labels,
    write_manifest,
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


def test_write_manifest_creates_parent_directories(tmp_path: Path) -> None:
    manifest = DownloadedAssetManifest(
        source_url="https://example.com/model.pt",
        downloaded_at=datetime.now(tz=timezone.utc),
        sha256="a" * 64,
        size_bytes=12,
    )
    manifest_path = tmp_path / "nested" / "asset.json"

    write_manifest(manifest_path, manifest)
    loaded = load_manifest(manifest_path)

    assert loaded == manifest


def test_load_manifest_returns_none_for_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("{broken", encoding="utf-8")

    assert load_manifest(manifest_path) is None


def test_download_url_to_path_writes_file_and_manifest(monkeypatch, tmp_path: Path) -> None:
    payload = b"hello-ultrasound"
    dest_path = tmp_path / "weights" / "model.pt"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ResponseStub([payload], content_length=len(payload)),
    )

    manifest = download_url_to_path("https://example.com/model.pt", dest_path, chunk_bytes=4)

    assert dest_path.read_bytes() == payload
    assert manifest.sha256 == hashlib.sha256(payload).hexdigest()
    assert manifest.size_bytes == len(payload)


def test_download_url_to_path_validates_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        download_url_to_path("https://example.com/file.bin", tmp_path / "x.bin", timeout_seconds=0)

    with pytest.raises(ValueError, match="chunk_bytes must be positive"):
        download_url_to_path("https://example.com/file.bin", tmp_path / "x.bin", chunk_bytes=0)


def test_download_url_to_path_cleans_temp_file_on_failure(monkeypatch, tmp_path: Path) -> None:
    dest_path = tmp_path / "artifacts" / "model.pt"

    def _fail(*_args: object, **_kwargs: object) -> _ResponseStub:
        raise RuntimeError("network down")

    monkeypatch.setattr("urllib.request.urlopen", _fail)

    with pytest.raises(RuntimeError, match="network down"):
        download_url_to_path("https://example.com/model.pt", dest_path)

    assert not dest_path.exists()
    assert not dest_path.with_suffix(".pt.tmp").exists()


def test_mask_to_xyxy_rejects_non_2d_masks() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        mask_to_xyxy(np.zeros((4, 4, 1), dtype=np.uint8))


def test_parse_yolo_txt_labels_rejects_invalid_numeric_fields() -> None:
    with pytest.raises(ValueError, match="Invalid numeric values"):
        parse_yolo_txt_labels("0 0.5 bad 0.2 0.2")


def test_parse_yolo_txt_labels_rejects_out_of_range_class_names() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_yolo_txt_labels("2 0.5 0.5 0.2 0.2", class_names=["liver", "mass"])


def test_dataset_preparer_rejects_missing_inputs(tmp_path: Path) -> None:
    preparer = YoloDatasetPreparer(
        source_images_dir=tmp_path / "images",
        annotations_csv=tmp_path / "annotations.csv",
        output_dir=tmp_path / "out",
        class_names=["liver"],
    )

    with pytest.raises(FileNotFoundError, match="source images directory"):
        preparer.prepare()

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="annotations CSV"):
        preparer.prepare()


def test_dataset_preparer_fails_when_all_images_are_unreadable(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "bad.png").write_text("not-an-image", encoding="utf-8")

    annotations_csv = tmp_path / "annotations.csv"
    annotations_csv.write_text(
        "image_id,x_min,y_min,x_max,y_max,class_id\nbad,1,2,10,12,0\n",
        encoding="utf-8",
    )

    preparer = YoloDatasetPreparer(
        source_images_dir=images_dir,
        annotations_csv=annotations_csv,
        output_dir=tmp_path / "out",
        class_names=["liver"],
    )

    with pytest.raises(ValueError, match="No readable images"):
        preparer.prepare()


def test_yolo_trainer_train_and_validate_use_stubbed_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "liver_detection"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    (weights_dir / "last.pt").write_bytes(b"last")

    train_calls: list[dict[str, object]] = []
    validate_calls: list[dict[str, object]] = []
    frozen_flags: list[bool] = []

    class _Parameter:
        def __init__(self) -> None:
            self.requires_grad = True

    shared_params = [_Parameter(), _Parameter()]

    class _FakeResults:
        def __init__(self) -> None:
            self.save_dir = run_dir
            self.results_dict = {"metrics/mAP50(B)": np.float32(0.9), "junk": "skip"}

    class _FakeValResults:
        class box:
            map50 = 0.8
            map = 0.7
            mp = 0.6
            mr = 0.5

    class _YOLOStub:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.model = SimpleNamespace(
                named_parameters=lambda: [(f"p{i}", param) for i, param in enumerate(shared_params)]
            )

        def train(self, **kwargs: object) -> _FakeResults:
            train_calls.append(kwargs)
            frozen_flags.extend(param.requires_grad for param in shared_params)
            return _FakeResults()

        def val(self, **kwargs: object) -> _FakeValResults:
            validate_calls.append(kwargs)
            return _FakeValResults()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = _YOLOStub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: /tmp\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['liver']\n")

    trainer = YoloTrainer()
    train_result = trainer.train(
        YoloTrainingConfig(
            dataset_yaml=dataset_yaml,
            pretrained_weights="stub.pt",
            project_dir=tmp_path / "runs",
            run_name="liver_detection",
            freeze_layers=1,
            device="cpu",
        )
    )
    validation_result = trainer.validate(train_result.best_weights or weights_dir / "best.pt", dataset_yaml)

    assert train_result.best_weights == weights_dir / "best.pt"
    assert train_result.last_weights == weights_dir / "last.pt"
    assert train_result.metrics["metrics/mAP50(B)"] == pytest.approx(0.9)
    assert train_calls and train_calls[0]["device"] == "cpu"
    assert frozen_flags[0] is False
    assert frozen_flags[1] is True
    assert validate_calls and validate_calls[0]["data"] == str(dataset_yaml)
    assert validation_result == {"map50": 0.8, "map": 0.7, "mp": 0.6, "mr": 0.5}


def test_yolo_trainer_extract_metrics_handles_none_and_box_fallback() -> None:
    assert YoloTrainer._extract_metrics(None) == {}

    results = SimpleNamespace(
        box=SimpleNamespace(map50=np.float64(0.4), map=np.float64(0.3), mp="bad", mr=0.2)
    )

    assert YoloTrainer._extract_metrics(results) == {"map50": 0.4, "map": 0.3, "mr": 0.2}
