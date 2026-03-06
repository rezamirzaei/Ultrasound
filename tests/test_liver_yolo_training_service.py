"""Tests for liver YOLO training orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

import ultrasound.api.services.liver_yolo_training_service as liver_training_module
from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import YoloTrainRequest
from ultrasound.api.services.liver_yolo_training_service import LiverYoloTrainingService
from ultrasound.api.services.service_errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    NotFoundError,
    ServiceError,
)
from ultrasound.api.services.yolo_trainer import YoloTrainingResult


class _TrainerStub:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.last_dataset_yaml: Path | None = None

    def train(self, config) -> YoloTrainingResult:  # noqa: ANN001
        self.last_dataset_yaml = Path(config.dataset_yaml)
        run_dir = self.project_root / "outputs" / "api" / "yolo_runs" / "liver_detection"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        return YoloTrainingResult(
            best_weights=best,
            last_weights=last,
            metrics={"map50": 0.8},
            epochs_completed=int(config.epochs),
            run_dir=run_dir,
        )


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = project_root / "ui"
    artifacts_dir = project_root / "outputs" / "api"
    for path in (busi_dir, ndt_dir, ui_dir, artifacts_dir):
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


def test_training_service_uses_synthetic_dataset_and_publishes_weights(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    trainer = _TrainerStub(config.project_root)
    service = LiverYoloTrainingService(config, trainer_factory=lambda: trainer)

    response = service.train(
        YoloTrainRequest(
            use_synthetic=True,
            synthetic_samples=10,
            epochs=3,
            batch_size=2,
            image_size=320,
        )
    )

    assert trainer.last_dataset_yaml is not None
    assert trainer.last_dataset_yaml.is_file()
    assert response.best_weights is not None
    assert (config.project_root / "models" / "liver_yolo_best.pt").is_file()
    assert response.metrics["map50"] == pytest.approx(0.8)


def test_training_service_requires_real_dataset_when_not_synthetic(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = LiverYoloTrainingService(config, trainer_factory=lambda: _TrainerStub(config.project_root))

    with pytest.raises(NotFoundError):
        service.train(YoloTrainRequest(use_synthetic=False))


def test_training_service_accepts_ready_real_dataset_without_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    trainer = _TrainerStub(config.project_root)
    service = LiverYoloTrainingService(config, trainer_factory=lambda: trainer)
    dataset_root = config.data_dir / "liver_ultrasound_detection"
    images_flat = dataset_root / "images_flat"
    images_flat.mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations.csv").write_text("image_id,class_id,xmin,ymin,xmax,ymax\n", encoding="utf-8")

    prepare_calls: list[Path] = []

    class _PreparerStub:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            prepare_calls.append(Path(kwargs["source_images_dir"]))

        def prepare(self) -> Path:
            data_yaml = dataset_root / "data.yaml"
            data_yaml.write_text("path: .\n", encoding="utf-8")
            return data_yaml

    monkeypatch.setattr(liver_training_module, "YoloDatasetPreparer", _PreparerStub)
    response = service.train(YoloTrainRequest(use_synthetic=False, epochs=2, batch_size=2))

    assert prepare_calls == [images_flat]
    assert response.epochs_completed == 2


def test_training_service_maps_dataset_preparation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    service = LiverYoloTrainingService(config, trainer_factory=lambda: _TrainerStub(config.project_root))

    class _FailingPreparer:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            pass

        def prepare(self) -> Path:
            raise ValueError("bad annotations")

    monkeypatch.setattr(liver_training_module, "YoloDatasetPreparer", _FailingPreparer)
    with pytest.raises(InvalidRequestError, match="Dataset preparation failed: bad annotations"):
        service.train(YoloTrainRequest(use_synthetic=True))


def test_training_service_maps_runtime_dependency_failures(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    class _RuntimeFailingTrainer:
        def train(self, config) -> YoloTrainingResult:  # noqa: ANN001
            raise RuntimeError("ultralytics missing")

    service = LiverYoloTrainingService(config, trainer_factory=lambda: _RuntimeFailingTrainer())

    with pytest.raises(DependencyUnavailableError, match="ultralytics missing"):
        service.train(YoloTrainRequest(use_synthetic=True))


def test_training_service_maps_generic_training_failures(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    class _GenericFailingTrainer:
        def train(self, config) -> YoloTrainingResult:  # noqa: ANN001
            raise ValueError("boom")

    service = LiverYoloTrainingService(config, trainer_factory=lambda: _GenericFailingTrainer())

    with pytest.raises(ServiceError, match="Training failed: boom"):
        service.train(YoloTrainRequest(use_synthetic=True))


def test_publish_best_weights_returns_none_for_missing_weights(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    service = LiverYoloTrainingService(config, trainer_factory=lambda: _TrainerStub(config.project_root))

    assert service._publish_best_weights(config.project_root / "missing.pt") is None
    assert service._publish_best_weights(None) is None
