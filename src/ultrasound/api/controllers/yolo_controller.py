"""YOLO endpoints for liver ultrasound detection and general YOLO status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.schemas import (
    LiverSampleResponse,
    LiverYoloLabStatusResponse,
    YoloPredictRequest,
    YoloPredictResponse,
    YoloStatusResponse,
)
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(
    tags=["yolo"],
    dependencies=[Depends(require_role("viewer"))],
)


@router.get("/yolo/status", response_model=YoloStatusResponse)
def yolo_status(
    container: ApplicationContainer = Depends(get_container),
) -> YoloStatusResponse:
    """Report whether YOLO backend dependencies are available."""
    return container.yolo_service.status()


@router.get("/yolo/liver/status", response_model=LiverYoloLabStatusResponse)
def liver_lab_status(
    container: ApplicationContainer = Depends(get_container),
) -> LiverYoloLabStatusResponse:
    """Combined YOLO backend + liver dataset readiness."""
    return container.liver_yolo_lab_service.lab_status()


@router.get(
    "/yolo/liver/samples/{category}/{sample_index}",
    response_model=LiverSampleResponse,
)
def get_liver_sample(
    category: str,
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> LiverSampleResponse:
    """Load a liver ultrasound sample with bounding-box annotations."""
    try:
        return container.liver_yolo_lab_service.get_sample(category, sample_index)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/yolo/liver/samples/{category}/{sample_index}/predict",
    response_model=YoloPredictResponse,
)
def predict_liver_sample(
    category: str,
    sample_index: int,
    request: YoloPredictRequest,
    container: ApplicationContainer = Depends(get_container),
) -> YoloPredictResponse:
    """Run YOLO inference on a liver ultrasound sample."""
    try:
        return container.liver_yolo_lab_service.predict(category, sample_index, request)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
