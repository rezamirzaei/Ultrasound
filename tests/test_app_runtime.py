"""Focused runtime tests for app middleware and exception handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import ultrasound.api.app as app_module
from ultrasound.api.config import AppConfig


class _JobQueueServiceStub:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _ObservabilityServiceStub:
    METRICS_CONTENT_TYPE = "text/plain; version=0.0.4"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.exceptions: list[dict[str, str]] = []

    def observe_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self.requests.append(
            {
                "method": method,
                "route": route,
                "status_code": status_code,
                "duration_seconds": duration_seconds,
            }
        )

    def observe_http_exception(self, *, route: str, exception_type: str) -> None:
        self.exceptions.append({"route": route, "exception_type": exception_type})

    def render_metrics(self) -> str:
        return "inphase_requests_total 1\n"


class _ErrorAnalyticsServiceStub:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_error(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        detail: str,
        role: str | None,
    ) -> None:
        self.events.append(
            {
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "detail": detail,
                "role": role,
            }
        )


class _AuthServiceStub:
    def verify_token(self, token: str):  # type: ignore[no-untyped-def]
        if token == "viewer-token":
            return type("AuthSession", (), {"role": "viewer"})()
        raise ValueError("invalid token")


class _ApplicationContainerStub:
    def __init__(self, _config: AppConfig) -> None:
        self.job_queue_service = _JobQueueServiceStub()
        self.observability_service = _ObservabilityServiceStub()
        self.error_analytics_service = _ErrorAnalyticsServiceStub()
        self.auth_service = _AuthServiceStub()


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = project_root / "ui"
    artifacts_dir = project_root / "artifacts"
    for path in (busi_dir, ndt_dir, ui_dir / "app" / "views", artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<html><body>home</body></html>", encoding="utf-8")
    (ui_dir / "app" / "views" / "test.html").write_text(
        "<html><body>fragment</body></html>",
        encoding="utf-8",
    )
    return AppConfig(
        project_root=project_root,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
        database_url="sqlite:///:memory:",
    )


def test_create_app_runtime_middleware_and_handlers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "ApplicationContainer", _ApplicationContainerStub)
    app = app_module.create_app(config=_make_config(tmp_path))

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/teapot")
    def teapot() -> None:
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    @app.get("/validate/{value}")
    def validate(value: int) -> dict[str, int]:
        return {"value": value}

    container = app.state.container

    with TestClient(app, raise_server_exceptions=False) as client:
        ok_response = client.get("/ok")
        assert ok_response.status_code == 200
        assert ok_response.headers["X-Request-ID"]

        ui_response = client.get("/ui/app/views/test.html")
        assert ui_response.status_code == 200
        assert ui_response.headers["Cache-Control"] == "no-store, max-age=0"
        assert "etag" not in ui_response.headers
        assert "last-modified" not in ui_response.headers
        assert ui_response.headers["X-Request-ID"]

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert metrics_response.text == "inphase_requests_total 1\n"
        assert metrics_response.headers["content-type"].startswith(
            container.observability_service.METRICS_CONTENT_TYPE
        )

        http_error = client.get("/teapot", headers={"Authorization": "Bearer viewer-token"})
        assert http_error.status_code == 418
        assert http_error.json()["code"] == "http_exception"
        assert http_error.json()["detail"] == "teapot"
        assert http_error.headers["X-Request-ID"] == http_error.json()["request_id"]

        validation_error = client.get("/validate/not-an-int")
        assert validation_error.status_code == 422
        assert validation_error.json()["code"] == "validation_error"
        assert validation_error.headers["X-Request-ID"] == validation_error.json()["request_id"]

        internal_error = client.get("/boom")
        assert internal_error.status_code == 500
        assert internal_error.json()["code"] == "internal_error"
        assert internal_error.headers["X-Request-ID"] == internal_error.json()["request_id"]

    assert container.job_queue_service.start_calls == 1
    assert container.job_queue_service.stop_calls == 1
    assert any(event["path"] == "/teapot" and event["role"] == "viewer" for event in container.error_analytics_service.events)
    assert any(event["path"] == "/boom" and event["status_code"] == 500 for event in container.error_analytics_service.events)
    assert any(entry["route"] == "/ok" and entry["status_code"] == 200 for entry in container.observability_service.requests)
    assert any(entry["route"] == "/teapot" and entry["status_code"] == 418 for entry in container.observability_service.requests)
    assert any(entry["route"] == "/boom" and entry["exception_type"] == "RuntimeError" for entry in container.observability_service.exceptions)


def test_app_module_getattr_caches_lazy_app(monkeypatch) -> None:
    sentinel = object()
    cast(dict[str, object], app_module.__dict__).pop("app", None)
    monkeypatch.setattr(app_module, "create_app", lambda: sentinel)

    resolved = app_module.__getattr__("app")

    assert resolved is sentinel
    assert app_module.app is sentinel
    with pytest.raises(AttributeError):
        app_module.__getattr__("missing")
