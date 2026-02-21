"""FastAPI application factory for REST API and AngularJS UI."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ultrasound.api.config import AppConfig
from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers import (
    auth_router,
    dashboard_router,
    health_router,
    ops_router,
    preprocessing_router,
)
from ultrasound.api.models.schemas import ApiError


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_config = config or AppConfig.from_project_root()
    container = ApplicationContainer(resolved_config)

    app = FastAPI(
        title="Ultrasound Imaging Toolkit API",
        version="1.0.0",
        description="REST API + AngularJS MVC UI for ultrasound analytics workflows.",
    )

    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = "/api/v1"
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(dashboard_router, prefix=api_prefix)
    app.include_router(preprocessing_router, prefix=api_prefix)
    app.include_router(ops_router, prefix=api_prefix)

    def _request_id(request: Request) -> str:
        return str(getattr(request.state, "request_id", ""))

    def _resolve_role(request: Request) -> str | None:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None

        token = auth_header.split(" ", 1)[1].strip()
        try:
            return request.app.state.container.auth_service.verify_token(token).role
        except Exception:
            return None

    def _record_error(request: Request, status_code: int, detail: str) -> None:
        container_obj = getattr(request.app.state, "container", None)
        if container_obj is None:
            return
        container_obj.error_analytics_service.record_error(
            request_id=_request_id(request) or "unknown",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            detail=detail,
            role=_resolve_role(request),
        )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = _request_id(request)
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = str(exc.detail)
        _record_error(request, exc.status_code, detail)
        payload = ApiError(
            detail=detail,
            request_id=_request_id(request) or None,
            status_code=exc.status_code,
            code="http_exception",
        )
        response = JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(exclude_none=True),
        )
        response.headers["X-Request-ID"] = _request_id(request)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        detail = "Invalid request payload"
        _record_error(request, 422, detail)
        payload = ApiError(
            detail=detail,
            request_id=_request_id(request) or None,
            status_code=422,
            code="validation_error",
        )
        response = JSONResponse(status_code=422, content=payload.model_dump(exclude_none=True))
        response.headers["X-Request-ID"] = _request_id(request)
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        detail = "Internal server error"
        _record_error(request, 500, str(exc))
        payload = ApiError(
            detail=detail,
            request_id=_request_id(request) or None,
            status_code=500,
            code="internal_error",
        )
        response = JSONResponse(status_code=500, content=payload.model_dump(exclude_none=True))
        response.headers["X-Request-ID"] = _request_id(request)
        return response

    resolved_config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if resolved_config.ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=resolved_config.ui_dir, html=True), name="ui")

    app.mount(
        "/artifacts",
        StaticFiles(directory=resolved_config.artifacts_dir, html=False),
        name="artifacts",
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/index.html")

    return app


app = create_app()
