"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ultrasound import __version__
from ultrasound.api.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness probe for API and UI clients."""
    return HealthResponse(
        status="ok",
        message="Ultrasound API is running",
        version=__version__,
    )
