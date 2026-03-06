"""Tests for public package exports and lazy API entrypoints."""

from __future__ import annotations

import importlib

import ultrasound


def test_ultrasound_public_api_exports() -> None:
    assert "UNet" in ultrasound.__all__
    assert "UltrasoundClassifier" in ultrasound.__all__
    assert "visualize_results" in ultrasound.__all__
    assert ultrasound.UNet.__name__ == "UNet"
    assert ultrasound.UltrasoundClassifier.__name__ == "UltrasoundClassifier"


def test_api_main_resolves_lazy_app(monkeypatch) -> None:
    api_app = importlib.import_module("ultrasound.api.app")
    api_main = importlib.reload(importlib.import_module("ultrasound.api.main"))
    sentinel = object()

    monkeypatch.setattr(api_app, "app", sentinel, raising=False)
    api_main.__dict__.pop("app", None)

    assert api_main.app is sentinel
    assert api_main.app is sentinel
    assert "app" in dir(api_main)
    assert "create_app" in dir(api_main)


def test_api_package_resolves_lazy_create_app(monkeypatch) -> None:
    api_pkg = importlib.reload(importlib.import_module("ultrasound.api"))
    api_app = importlib.import_module("ultrasound.api.app")

    sentinel = object()
    monkeypatch.setattr(api_app, "create_app", sentinel, raising=False)
    api_pkg.__dict__.pop("create_app", None)

    assert api_pkg.create_app is sentinel
    assert api_pkg.create_app is sentinel
    assert "create_app" in dir(api_pkg)


def test_api_package_invalid_attribute_raises() -> None:
    api_pkg = importlib.reload(importlib.import_module("ultrasound.api"))

    try:
        _ = api_pkg.not_a_real_symbol
    except AttributeError as exc:
        assert "not_a_real_symbol" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected AttributeError")
