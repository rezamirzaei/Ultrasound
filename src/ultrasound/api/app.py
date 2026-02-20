"""FastAPI application factory for REST API and AngularJS UI."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ultrasound.api.config import AppConfig
from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers import dashboard_router, health_router, preprocessing_router


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
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(dashboard_router, prefix=api_prefix)
    app.include_router(preprocessing_router, prefix=api_prefix)

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
