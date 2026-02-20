"""Shared dependency providers for API controllers."""

from __future__ import annotations

from fastapi import Request

from ultrasound.api.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    """Resolve application container from app state."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Application container is not configured")
    return container
