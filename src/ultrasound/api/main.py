"""ASGI entrypoint for the ultrasound API."""

from ultrasound.api.app import app, create_app

__all__ = ["app", "create_app"]
