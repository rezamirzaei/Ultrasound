"""Tests for BUSI and liver YOLO lab services."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ultrasound.api.services.busi_yolo_lab_service as busi_yolo_module
from ultrasound.api.config import AppConfig
from ultrasound.api.models.domain import BusiSampleRecord
from ultrasound.api.models.schemas import (
    DownloadedAssetManifest,
    YoloPredictRequest,
    YoloPredictResponse,
    YoloStatusResponse,
)
from ultrasound.api.services.busi_yolo_lab_service import BusiYoloLabService
from ultrasound.api.services.liver_yolo_lab_service import LiverYoloLabService
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.service_errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    NotFoundError,
)
from ultrasound.api.services.yolo_utils import write_manifest
from ultrasound.data.liver_dataset import create_synthetic_liver_dataset


class _RecordingYoloService:
    DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = ("yolo11n.pt", "yolov8n.pt")

    def __init__(self) -> None:
        self.last_request: YoloPredictRequest | None = None
        self.last_image_shape: tuple[int, ...] | None = None

    def status(self) -> YoloStatusResponse:
        return YoloStatusResponse(
            available=True,
            backend="ultralytics",
            default_models=list(self.DEFAULT_MODEL_CANDIDATES),
        )

    def predict(self, image_rgb: np.ndarray, request: YoloPredictRequest) -> YoloPredictResponse:
        self.last_request = request
        self.last_image_shape = tuple(image_rgb.shape)
        return YoloPredictResponse(
            generated_at=datetime.now(tz=timezone.utc),
            model=request.model,
            image_shape=list(image_rgb.shape),
            detections=[],
            annotated_image_data_url=MediaService().as_png_data_url(image_rgb),
            backend="ultralytics",
        )


class _NoDefaultYoloService(_RecordingYoloService):
    DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = ()


class _FailingYoloService(_RecordingYoloService):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def predict(self, image_rgb: np.ndarray, request: YoloPredictRequest) -> YoloPredictResponse:
        raise self.error


class _DatasetRepositoryStub:
    def get_busi_sample(self, class_name: str, index: int) -> BusiSampleRecord:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 10:22] = 255
        return BusiSampleRecord(
            class_name=class_name,
            requested_index=index,
            resolved_index=index,
            total_samples=4,
            image_path=Path("/tmp/sample.png"),
            image_rgb=image,
            mask=mask,
        )


class _MissingDatasetRepositoryStub:
    def get_busi_sample(self, class_name: str, index: int) -> BusiSampleRecord:
        raise FileNotFoundError(f"missing {class_name}:{index}")


class _InvalidDatasetRepositoryStub:
    def get_busi_sample(self, class_name: str, index: int) -> BusiSampleRecord:
        raise ValueError(f"invalid {class_name}:{index}")


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    artifacts_dir = project_root / "outputs" / "api"
    ui_dir = project_root / "ui"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    busi_dir = data_dir / "busi"
    for path in (artifacts_dir, ui_dir, ndt_dir, busi_dir):
        path.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project_root,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
        database_url="sqlite:///:memory:",
    )


def test_busi_predict_uses_downloaded_recommended_model(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    yolo_service = _RecordingYoloService()
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=yolo_service,
    )
    recommended = service._model_path()
    recommended.write_bytes(b"weights")

    service.predict("benign", 0, YoloPredictRequest(model="yolov8n.pt"))

    assert yolo_service.last_request is not None
    assert yolo_service.last_request.model == str(recommended)


def test_busi_predict_rejects_missing_explicit_recommended_model(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )

    try:
        service.predict("benign", 0, YoloPredictRequest(model=str(service._model_path())))
    except InvalidRequestError as exc:
        assert "not downloaded yet" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected InvalidRequestError")


def test_busi_default_model_falls_back_to_builtin_when_candidates_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_NoDefaultYoloService(),
    )

    assert service.default_model == "yolo11n.pt"


def test_busi_model_status_uses_manifest_metadata(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )
    model_path = service._model_path()
    model_path.write_bytes(b"weights")
    write_manifest(
        service._manifest_path(),
        DownloadedAssetManifest(
            source_url=service.RECOMMENDED_MODEL_SOURCE_URL,
            downloaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            sha256="a" * 64,
            size_bytes=7,
        ),
    )

    status = service.model_status()

    assert status.downloaded is True
    assert status.sha256 == "a" * 64
    assert status.size_bytes == len(b"weights")


def test_busi_download_recommended_model_persists_manifest(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )

    def _download(_url: str, dest_path: Path) -> DownloadedAssetManifest:
        dest_path.write_bytes(b"weights")
        return DownloadedAssetManifest(
            source_url=service.RECOMMENDED_MODEL_SOURCE_URL,
            downloaded_at=datetime.now(tz=timezone.utc),
            sha256="b" * 64,
            size_bytes=7,
        )

    monkeypatch.setattr(busi_yolo_module, "download_url_to_path", _download)

    status = service.download_recommended_model()

    assert status.downloaded is True
    assert status.sha256 == "b" * 64
    assert service._manifest_path().exists()


def test_busi_download_recommended_model_maps_failures(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )
    monkeypatch.setattr(
        busi_yolo_module,
        "download_url_to_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(DependencyUnavailableError, match="Could not download recommended BUSI YOLO weights"):
        service.download_recommended_model(force=True)


def test_busi_get_sample_leaves_normal_masks_unlabeled(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )

    sample = service.get_sample("normal", 0)

    assert sample.sample.class_name == "normal"
    assert sample.bbox_xyxy is not None
    assert sample.yolo_labels == []
    assert sample.raw_yolo_labels == ""


@pytest.mark.parametrize(
    ("repository", "error_type"),
    [
        (_MissingDatasetRepositoryStub(), NotFoundError),
        (_InvalidDatasetRepositoryStub(), InvalidRequestError),
    ],
)
def test_busi_sample_loading_maps_repository_errors(
    tmp_path: Path,
    repository: Any,
    error_type: type[Exception],
) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=repository,
        media_service=MediaService(),
        yolo_service=_RecordingYoloService(),
    )

    with pytest.raises(error_type):
        service.get_sample("benign", 0)


def test_busi_predict_maps_runtime_failures(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_FailingYoloService(RuntimeError("ultralytics missing")),
    )

    with pytest.raises(DependencyUnavailableError, match="ultralytics missing"):
        service.predict("benign", 0, YoloPredictRequest(model="yolov8n.pt"))


def test_busi_predict_maps_value_failures(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = BusiYoloLabService(
        config=config,
        dataset_repository=_DatasetRepositoryStub(),
        media_service=MediaService(),
        yolo_service=_FailingYoloService(ValueError("invalid request")),
    )

    with pytest.raises(InvalidRequestError, match="invalid request"):
        service.predict("benign", 0, YoloPredictRequest(model="yolov8n.pt"))


def test_liver_service_accepts_case_insensitive_category_and_synthetic_dataset(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    create_synthetic_liver_dataset(config.data_dir / "liver_ultrasound_detection", n_samples=3)
    yolo_service = _RecordingYoloService()
    service = LiverYoloLabService(config, MediaService(), yolo_service)

    sample = service.get_sample("benign", 0)
    prediction = service.predict("BENIGN", 0, YoloPredictRequest(model="yolov8n.pt"))

    assert sample.category == "Benign"
    assert prediction.image_shape[2] == 3
    assert yolo_service.last_image_shape is not None


def test_liver_service_prefers_trained_weights_when_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    create_synthetic_liver_dataset(config.data_dir / "liver_ultrasound_detection", n_samples=2)
    trained = config.project_root / "models" / "liver_yolo_best.pt"
    trained.parent.mkdir(parents=True, exist_ok=True)
    trained.write_bytes(b"weights")

    yolo_service = _RecordingYoloService()
    service = LiverYoloLabService(config, MediaService(), yolo_service)

    service.predict("benign", 0, YoloPredictRequest(model="yolov8n.pt"))

    assert yolo_service.last_request is not None
    assert yolo_service.last_request.model == str(trained)
