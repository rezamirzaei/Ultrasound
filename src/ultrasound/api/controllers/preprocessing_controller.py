"""Preprocessing endpoints for interactive UI previews."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container
from ultrasound.api.models.schemas import PreprocessingPreviewResponse, PreprocessingRequest

router = APIRouter(tags=["preprocessing"])


@router.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
def preview_preprocessing(
    request: PreprocessingRequest,
    container: ApplicationContainer = Depends(get_container),
) -> PreprocessingPreviewResponse:
    """Run preprocessing methods and return images/metrics for UI rendering."""
    try:
        return container.preprocessing_service.preview(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
