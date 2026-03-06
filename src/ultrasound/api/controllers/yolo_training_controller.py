"""YOLO training endpoints for liver ultrasound detection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import (
    LiverDatasetStatusResponse,
    YoloTrainRequest,
    YoloTrainResponse,
)
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(
    tags=["yolo-training"],
    dependencies=[Depends(require_role("viewer"))],
)


# -- Endpoints ---------------------------------------------------------------

@router.get("/yolo/liver/dataset/status", response_model=LiverDatasetStatusResponse)
def liver_dataset_status(
    container: ApplicationContainer = Depends(get_container),
) -> LiverDatasetStatusResponse:
    """Check if the liver ultrasound detection dataset is available."""
    return container.liver_yolo_lab_service.dataset_status()


@router.post("/yolo/liver/train", response_model=YoloTrainResponse)
def train_liver_yolo(
    request: YoloTrainRequest,
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> YoloTrainResponse:
    """Launch a YOLO training run for liver ultrasound detection."""
    try:
        return container.liver_yolo_training_service.train(request)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


