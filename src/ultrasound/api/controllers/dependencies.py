"""Shared dependency providers for API controllers."""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.models.domain import AuthSessionRecord

bearer_scheme = HTTPBearer(auto_error=False)


def get_container(request: Request) -> ApplicationContainer:
    """Resolve application container from app state."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Application container is not configured")
    return container


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    container: ApplicationContainer = Depends(get_container),
) -> AuthSessionRecord:
    """Resolve current user from bearer token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        return container.auth_service.verify_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def require_role(
    min_role: str,
) -> Callable[[AuthSessionRecord, ApplicationContainer], AuthSessionRecord]:
    """Create dependency that enforces minimum role access."""

    def dependency(
        current_user: AuthSessionRecord = Depends(get_current_user),
        container: ApplicationContainer = Depends(get_container),
    ) -> AuthSessionRecord:
        if not container.auth_service.has_role(current_user.role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{min_role}' or higher is required",
            )
        return current_user

    return dependency
