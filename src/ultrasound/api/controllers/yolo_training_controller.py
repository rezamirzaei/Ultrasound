"""YOLO training endpoints for liver ultrasound detection."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.services.liver_dataset import (
    CLASS_NAMES,
    create_synthetic_liver_dataset,
    resolve_liver_paths,
    summarize_dataset,
)
from ultrasound.api.services.yolo_trainer import (
    YoloDatasetPreparer,
    YoloTrainer,
    YoloTrainingConfig,
)

router = APIRouter(
    tags=["yolo-training"],
    dependencies=[Depends(require_role("viewer"))],
)


# -- Request / Response schemas ---------------------------------------------

class LiverDatasetStatusResponse(BaseModel):
    ready: bool
    summary: dict
    generated_at: datetime


class YoloTrainRequest(BaseModel):
    pretrained_weights: str = Field(default="yolo11n.pt", min_length=1)
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=128)
    image_size: int = Field(default=640, ge=160, le=2048)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    patience: int = Field(default=10, ge=1, le=100)
    train_ratio: float = Field(default=0.8, gt=0.1, lt=1.0)
    freeze_layers: int = Field(default=0, ge=0, le=50)
    use_synthetic: bool = Field(default=False)
    synthetic_samples: int = Field(default=30, ge=10, le=500)


class YoloTrainResponse(BaseModel):
    generated_at: datetime
    best_weights: str | None = None
    last_weights: str | None = None
    epochs_completed: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    run_dir: str | None = None


# -- Endpoints ---------------------------------------------------------------

@router.get("/yolo/liver/dataset/status", response_model=LiverDatasetStatusResponse)
def liver_dataset_status(
    container: ApplicationContainer = Depends(get_container),
) -> LiverDatasetStatusResponse:
    """Check if the liver ultrasound detection dataset is available."""
    paths = resolve_liver_paths(container.config.data_dir)
    summary = summarize_dataset(paths) if paths.is_ready else {"status": "not_found"}
    return LiverDatasetStatusResponse(
        ready=paths.is_ready,
        summary=summary,
        generated_at=datetime.now(tz=timezone.utc),
    )


@router.post("/yolo/liver/train", response_model=YoloTrainResponse)
def train_liver_yolo(
    request: YoloTrainRequest,
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> YoloTrainResponse:
    """Launch a YOLO training run for liver ultrasound detection."""
    data_dir = container.config.data_dir / "liver_ultrasound_detection"
    yolo_dir = container.config.data_dir / "liver_yolo"
    output_dir = container.config.artifacts_dir / "yolo_runs"

    # Get or create dataset
    if request.use_synthetic:
        paths = create_synthetic_liver_dataset(data_dir, n_samples=request.synthetic_samples)
    else:
        paths = resolve_liver_paths(container.config.data_dir)
        if not paths.is_ready:
            raise HTTPException(
                status_code=404,
                detail="Liver dataset not found. Download it first or set use_synthetic=true.",
            )

    # Prepare YOLO format
    try:
        preparer = YoloDatasetPreparer(
            source_images_dir=paths.train_images_dir,
            annotations_csv=paths.annotations_csv,
            output_dir=yolo_dir,
            class_names=CLASS_NAMES,
            train_ratio=request.train_ratio,
        )
        data_yaml = preparer.prepare()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dataset preparation failed: {exc}") from exc

    # Train
    try:
        config = YoloTrainingConfig(
            dataset_yaml=data_yaml,
            pretrained_weights=request.pretrained_weights,
            epochs=request.epochs,
            image_size=request.image_size,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            patience=request.patience,
            project_dir=output_dir,
            run_name="liver_detection",
            freeze_layers=request.freeze_layers,
        )
        trainer = YoloTrainer()
        result = trainer.train(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return YoloTrainResponse(
        generated_at=datetime.now(tz=timezone.utc),
        best_weights=str(result.best_weights) if result.best_weights else None,
        last_weights=str(result.last_weights) if result.last_weights else None,
        epochs_completed=result.epochs_completed,
        metrics=result.metrics,
        run_dir=str(result.run_dir) if result.run_dir else None,
    )


