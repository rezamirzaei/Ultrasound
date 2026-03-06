"""Preprocessing endpoints for interactive UI previews."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.controllers.error_mapping import raise_http_error
from ultrasound.api.models.schemas import PreprocessingPreviewResponse, PreprocessingRequest
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(
    tags=["preprocessing"],
    dependencies=[Depends(require_role("analyst"))],
)


@router.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
def preview_preprocessing(
    request: PreprocessingRequest,
    container: ApplicationContainer = Depends(get_container),
) -> PreprocessingPreviewResponse:
    """Run preprocessing methods and return images/metrics for UI rendering."""
    try:
        return container.preprocessing_service.preview(request)
    except ServiceError as exc:
        raise_http_error(exc)
