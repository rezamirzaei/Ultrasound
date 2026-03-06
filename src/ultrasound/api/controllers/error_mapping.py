"""Shared translation from service-layer exceptions to HTTP responses."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException

from ultrasound.api.services.service_errors import ServiceError


def raise_http_error(exc: ServiceError) -> NoReturn:
    """Re-raise a service error as a FastAPI HTTPException."""
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
