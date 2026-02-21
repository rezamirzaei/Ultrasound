"""Operational diagnostics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.schemas import OpsErrorEvent, OpsErrorSummaryResponse

router = APIRouter(
    tags=["ops"],
    dependencies=[Depends(require_role("admin"))],
)


@router.get("/ops/errors/summary", response_model=OpsErrorSummaryResponse)
def get_error_summary(
    window_minutes: int = Query(default=60, ge=1, le=7 * 24 * 60),
    container: ApplicationContainer = Depends(get_container),
) -> OpsErrorSummaryResponse:
    """Return aggregate API error counters over the requested time window."""
    payload = container.error_analytics_service.summary(window_minutes=window_minutes)
    return OpsErrorSummaryResponse(**payload)


@router.get("/ops/errors/recent", response_model=list[OpsErrorEvent])
def get_recent_errors(
    limit: int = Query(default=20, ge=1, le=200),
    container: ApplicationContainer = Depends(get_container),
) -> list[OpsErrorEvent]:
    """Return most recent API errors for troubleshooting and triage."""
    events = container.error_analytics_service.recent_errors(limit=limit)
    return [OpsErrorEvent(**event.model_dump()) for event in events]
