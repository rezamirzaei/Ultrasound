"""YOLO endpoints for field-style inspection imagery."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import (
    FieldYoloMetadata,
    FieldYoloRecordDetail,
    FieldYoloRecordSummary,
    YoloPredictRequest,
    YoloPredictResponse,
    YoloStatusResponse,
)

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


@router.post("/yolo/field/upload", response_model=FieldYoloRecordSummary)
async def upload_field_record(
    metadata_json: str = Form(...),
    image: UploadFile = File(...),
    labels: UploadFile | None = File(default=None),
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> FieldYoloRecordSummary:
    """Upload one field-style image + metadata record (optional YOLO labels)."""
    try:
        metadata = FieldYoloMetadata.model_validate_json(metadata_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    image_blob = await image.read()
    if not image_blob:
        raise HTTPException(status_code=400, detail="Image file is empty")

    labels_blob = await labels.read() if labels is not None else None
    try:
        return container.field_yolo_service.create_record(
            metadata=metadata,
            image_filename=image.filename or "field_image",
            image_blob=image_blob,
            labels_filename=labels.filename if labels is not None else None,
            labels_blob=labels_blob,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/yolo/field/records", response_model=list[FieldYoloRecordSummary])
def list_field_records(
    limit: int = Query(default=50, ge=1, le=200),
    container: ApplicationContainer = Depends(get_container),
) -> list[FieldYoloRecordSummary]:
    """List recent field inspection records stored under artifacts/."""
    return container.field_yolo_service.list_records(limit=limit)


@router.get("/yolo/field/records/{record_id}", response_model=FieldYoloRecordDetail)
def get_field_record(
    record_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> FieldYoloRecordDetail:
    """Fetch one field record including image preview and parsed labels."""
    try:
        return container.field_yolo_service.get_record(record_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/yolo/field/records/{record_id}/predict", response_model=YoloPredictResponse)
def predict_field_record(
    record_id: str,
    request: YoloPredictRequest,
    container: ApplicationContainer = Depends(get_container),
) -> YoloPredictResponse:
    """Run YOLO inference on a stored field record image."""
    try:
        image_rgb = container.field_yolo_service.load_image_rgb(record_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return container.yolo_service.predict(image_rgb=image_rgb, request=request)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
