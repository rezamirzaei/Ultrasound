"""Liver-specific orchestration for YOLO dataset preparation and training."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import YoloTrainRequest, YoloTrainResponse
from ultrasound.api.services.service_errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    NotFoundError,
    ServiceError,
)
from ultrasound.api.services.yolo_trainer import (
    YoloDatasetPreparer,
    YoloTrainer,
    YoloTrainingConfig,
    YoloTrainingResult,
)
from ultrasound.data.liver_dataset import (
    CLASS_NAMES,
    LiverDatasetPaths,
    create_synthetic_liver_dataset,
    resolve_liver_paths,
)

logger = logging.getLogger("inphase.yolo.training")


class YoloTrainingBackend(Protocol):
    def train(self, config: YoloTrainingConfig) -> YoloTrainingResult: ...


class LiverYoloTrainingService:
    """Prepare liver datasets, launch YOLO training, and publish weights."""

    def __init__(
        self,
        config: AppConfig,
        trainer_factory: Callable[[], YoloTrainingBackend] | None = None,
    ) -> None:
        self._config = config
        self._trainer_factory = trainer_factory or YoloTrainer

    def train(self, request: YoloTrainRequest) -> YoloTrainResponse:
        paths = self._resolve_dataset(request)
        data_yaml = self._prepare_dataset(paths=paths, request=request)
        result = self._run_training(data_yaml=data_yaml, request=request)
        self._publish_best_weights(result.best_weights)
        return YoloTrainResponse(
            generated_at=datetime.now(tz=timezone.utc),
            best_weights=str(result.best_weights) if result.best_weights else None,
            last_weights=str(result.last_weights) if result.last_weights else None,
            epochs_completed=result.epochs_completed,
            metrics=result.metrics,
            run_dir=str(result.run_dir) if result.run_dir else None,
        )

    def _resolve_dataset(self, request: YoloTrainRequest) -> LiverDatasetPaths:
        data_dir = self._config.data_dir / "liver_ultrasound_detection"
        if request.use_synthetic:
            return create_synthetic_liver_dataset(data_dir, n_samples=request.synthetic_samples)

        paths = resolve_liver_paths(self._config.data_dir)
        if not paths.is_ready:
            raise NotFoundError(
                "Liver dataset not found. Download it first or set use_synthetic=true."
            )
        return paths

    def _prepare_dataset(self, paths: LiverDatasetPaths, request: YoloTrainRequest) -> Path:
        yolo_dir = self._config.data_dir / "liver_yolo"
        try:
            return YoloDatasetPreparer(
                source_images_dir=paths.train_images_dir,
                annotations_csv=paths.annotations_csv,
                output_dir=yolo_dir,
                class_names=CLASS_NAMES,
                train_ratio=request.train_ratio,
            ).prepare()
        except ServiceError:
            raise
        except Exception as exc:
            raise InvalidRequestError(f"Dataset preparation failed: {exc}") from exc

    def _run_training(self, data_yaml: Path, request: YoloTrainRequest) -> YoloTrainingResult:
        trainer = self._trainer_factory()
        config = YoloTrainingConfig(
            dataset_yaml=data_yaml,
            pretrained_weights=request.pretrained_weights,
            epochs=request.epochs,
            image_size=request.image_size,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            patience=request.patience,
            project_dir=self._config.artifacts_dir / "yolo_runs",
            run_name="liver_detection",
            freeze_layers=request.freeze_layers,
        )
        try:
            return trainer.train(config)
        except RuntimeError as exc:
            raise DependencyUnavailableError(str(exc)) from exc
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"Training failed: {exc}") from exc

    def _publish_best_weights(self, best_weights: Path | None) -> Path | None:
        if best_weights is None or not best_weights.exists():
            return None

        artifact_dir = self._config.project_root / "models"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "liver_yolo_best.pt"
        shutil.copy2(best_weights, destination)
        logger.info("Copied best weights to %s", destination)
        return destination
