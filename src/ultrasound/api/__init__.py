"""Ultrasound web API package."""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from ultrasound.api.app import app

        return app
    if name == "create_app":
        from ultrasound.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
