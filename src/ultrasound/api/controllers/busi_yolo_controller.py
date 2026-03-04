"""YOLO endpoints tailored for BUSI (medical ultrasound) samples."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import (
    BusiYoloLabStatusResponse,
    BusiYoloModelStatus,
    BusiYoloSampleResponse,
    YoloPredictRequest,
    YoloPredictResponse,
)

router = APIRouter(
    tags=["yolo-ultrasound"],
    dependencies=[Depends(require_role("viewer"))],
)


@router.get("/yolo/ultrasound/busi/status", response_model=BusiYoloLabStatusResponse)
def busi_yolo_status(
    container: ApplicationContainer = Depends(get_container),
) -> BusiYoloLabStatusResponse:
    """Report backend availability and recommended ultrasound YOLO weights."""
    return container.busi_yolo_lab_service.status()


@router.post("/yolo/ultrasound/busi/model/download", response_model=BusiYoloModelStatus)
def download_busi_yolo_model(
    force: bool = Query(default=False),
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> BusiYoloModelStatus:
    """Download the recommended BUSI YOLO weights (analyst-only)."""
    try:
        return container.busi_yolo_lab_service.download_recommended_model(force=force)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model download failed: {exc}") from exc


@router.get(
    "/yolo/ultrasound/busi/samples/{class_name}/{sample_index}",
    response_model=BusiYoloSampleResponse,
)
def get_busi_yolo_sample(
    class_name: Literal["benign", "malignant", "normal"],
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> BusiYoloSampleResponse:
    """Return one BUSI sample with derived YOLO labels from its segmentation mask."""
    try:
        return container.busi_yolo_lab_service.get_sample(class_name, sample_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/yolo/ultrasound/busi/samples/{class_name}/{sample_index}/predict",
    response_model=YoloPredictResponse,
)
def predict_busi_yolo_sample(
    class_name: Literal["benign", "malignant", "normal"],
    sample_index: int,
    request: YoloPredictRequest,
    container: ApplicationContainer = Depends(get_container),
) -> YoloPredictResponse:
    """Run YOLO inference on a BUSI sample image."""
    try:
        return container.busi_yolo_lab_service.predict(class_name, sample_index, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

