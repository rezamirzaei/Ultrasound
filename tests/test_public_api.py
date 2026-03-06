"""Tests for public package exports and lazy API entrypoints."""

from __future__ import annotations

import importlib
from pathlib import Path

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


def test_api_main_lazy_app_creates_sqlite_parent_dirs(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested_dir = project_root / "workspace" / "leaf"
    project_root.mkdir(parents=True, exist_ok=True)
    nested_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    (project_root / "src").mkdir(parents=True, exist_ok=True)

    database_path = project_root / "runtime" / "db" / "inphase.sqlite3"
    monkeypatch.chdir(nested_dir)
    monkeypatch.setenv("INPHASE_DATABASE_URL", f"sqlite:///{database_path}")

    api_app = importlib.reload(importlib.import_module("ultrasound.api.app"))
    api_main = importlib.reload(importlib.import_module("ultrasound.api.main"))
    api_app.__dict__.pop("app", None)
    api_main.__dict__.pop("app", None)

    resolved_app = api_main.app

    assert resolved_app.title == "Ultrasound Imaging Toolkit API"
    assert database_path.parent.exists()
    resolved_app.state.container.db.engine.dispose()


def test_api_package_resolves_lazy_create_app(monkeypatch) -> None:
    api_pkg = importlib.reload(importlib.import_module("ultrasound.api"))
    api_app = importlib.import_module("ultrasound.api.app")

    sentinel = object()
    monkeypatch.setattr(api_app, "create_app", sentinel, raising=False)
    api_pkg.__dict__.pop("create_app", None)

    assert api_pkg.create_app is sentinel
    assert api_pkg.create_app is sentinel
    assert "create_app" in dir(api_pkg)


def test_api_package_resolves_lazy_app(monkeypatch) -> None:
    api_pkg = importlib.reload(importlib.import_module("ultrasound.api"))
    api_app = importlib.import_module("ultrasound.api.app")
    sentinel = object()

    monkeypatch.setattr(api_app, "app", sentinel, raising=False)
    api_pkg.__dict__.pop("app", None)

    assert api_pkg.app is sentinel
    assert api_pkg.app is sentinel
    exported = dir(api_pkg)
    assert "app" in exported
    assert "create_app" in exported


def test_api_package_invalid_attribute_raises() -> None:
    api_pkg = importlib.reload(importlib.import_module("ultrasound.api"))

    try:
        _ = api_pkg.not_a_real_symbol
    except AttributeError as exc:
        assert "not_a_real_symbol" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected AttributeError")
