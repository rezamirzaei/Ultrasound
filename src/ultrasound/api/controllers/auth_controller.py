"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, get_current_user
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import AuthMeResponse, LoginRequest, LoginResponse

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    container: ApplicationContainer = Depends(get_container),
) -> LoginResponse:
    """Authenticate user credentials and issue bearer token."""
    try:
        session = container.auth_service.authenticate(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    return LoginResponse(
        access_token=container.auth_service.issue_token(session),
        token_type="Bearer",
        username=session.username,
        role=session.role,
        expires_at=session.expires_at,
    )


@router.get("/auth/me", response_model=AuthMeResponse)
def me(current_user: AuthSessionRecord = Depends(get_current_user)) -> AuthMeResponse:
    """Return current authenticated user profile."""
    return AuthMeResponse(
        username=current_user.username,
        role=current_user.role,
        expires_at=current_user.expires_at,
    )
