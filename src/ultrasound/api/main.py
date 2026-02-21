"""ASGI entrypoint for the ultrasound API."""

from __future__ import annotations

from typing import Any

from ultrasound.api.app import create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from ultrasound.api.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

