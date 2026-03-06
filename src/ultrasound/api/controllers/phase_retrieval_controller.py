"""Endpoints for real-data phase retrieval previews."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.controllers.error_mapping import raise_http_error
from ultrasound.api.models.schemas import (
    PhaseRetrievalPreviewRequest,
    PhaseRetrievalPreviewResponse,
    PhaseRetrievalStatusResponse,
)
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(
    tags=["phase_retrieval"],
    dependencies=[Depends(require_role("viewer"))],
)


@router.get("/phase-retrieval/picmus/status", response_model=PhaseRetrievalStatusResponse)
def get_picmus_phase_retrieval_status(
    container: ApplicationContainer = Depends(get_container),
) -> PhaseRetrievalStatusResponse:
    """Return PICMUS availability and tuned phase-retrieval defaults."""
    return container.phase_retrieval_service.get_status()


@router.post("/phase-retrieval/picmus/preview", response_model=PhaseRetrievalPreviewResponse)
def preview_picmus_phase_retrieval(
    request: PhaseRetrievalPreviewRequest,
    container: ApplicationContainer = Depends(get_container),
) -> PhaseRetrievalPreviewResponse:
    """Run tuned phase retrieval on a real PICMUS RF segment."""
    try:
        return container.phase_retrieval_service.preview(request)
    except ServiceError as exc:
        raise_http_error(exc)
