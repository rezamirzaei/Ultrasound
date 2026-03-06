"""ASGI entrypoint for the ultrasound API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ultrasound.api.app import create_app

__all__ = ["app", "create_app"]

if TYPE_CHECKING:
    from fastapi import FastAPI

    app: FastAPI


def __getattr__(name: str) -> Any:
    if name == "app":
        from ultrasound.api.app import app as resolved_app

        globals()["app"] = resolved_app
        return resolved_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
