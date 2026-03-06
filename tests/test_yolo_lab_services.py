"""Tests for BUSI and liver YOLO lab services."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ultrasound.api.config import AppConfig
from ultrasound.api.models.domain import BusiSampleRecord
from ultrasound.api.models.schemas import YoloPredictRequest, YoloPredictResponse, YoloStatusResponse
from ultrasound.api.services.busi_yolo_lab_service import BusiYoloLabService
from ultrasound.api.services.liver_yolo_lab_service import LiverYoloLabService
from ultrasound.api.services.media_service import MediaService
from ultrasound.data.liver_dataset import create_synthetic_liver_dataset


class _RecordingYoloService:
    DEFAULT_MODEL_CANDIDATES = ("yolo11n.pt", "yolov8n.pt")

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
    except ValueError as exc:
        assert "not downloaded yet" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


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
