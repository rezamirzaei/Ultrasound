"""Ultrasound web API package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["app", "create_app"]

if TYPE_CHECKING:
    from fastapi import FastAPI

    app: FastAPI


def __getattr__(name: str) -> Any:
    if name == "app":
        from ultrasound.api.app import app as resolved_app

        globals()["app"] = resolved_app
        return resolved_app
    if name == "create_app":
        from ultrasound.api.app import create_app as app_factory

        globals()["create_app"] = app_factory
        return app_factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
