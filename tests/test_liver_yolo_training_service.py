"""Tests for liver YOLO training orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import YoloTrainRequest
from ultrasound.api.services.liver_yolo_training_service import LiverYoloTrainingService
from ultrasound.api.services.service_errors import NotFoundError
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
