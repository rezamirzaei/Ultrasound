"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import bearer_scheme, get_container, get_current_user
from ultrasound.api.controllers.error_mapping import raise_http_error
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import (
    AuthMeResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
)
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    container: ApplicationContainer = Depends(get_container),
) -> LoginResponse:
    """Authenticate user credentials and issue bearer token."""
    try:
        session = container.auth_service.authenticate(request.username, request.password)
    except ServiceError as exc:
        raise_http_error(exc)

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


@router.post("/auth/logout", response_model=LogoutResponse)
def logout(
    current_user: AuthSessionRecord = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    container: ApplicationContainer = Depends(get_container),
) -> LogoutResponse:
    """Revoke current bearer token server-side."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    revoked = container.auth_service.revoke_token(credentials.credentials)
    return LogoutResponse(
        success=bool(revoked),
        username=current_user.username,
        revoked_token=bool(revoked),
    )
