"""Tests for the executable API entrypoint."""

from __future__ import annotations

import importlib


def test_main_runs_migrations_before_server(monkeypatch) -> None:
    module = importlib.reload(importlib.import_module("ultrasound.api.__main__"))
    migration_flags: list[bool] = []
    serve_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.delenv("INPHASE_SKIP_MIGRATIONS", raising=False)
    monkeypatch.setenv("INPHASE_MIGRATION_AUTO_STAMP", "yes")
    monkeypatch.setattr(
        module,
        "upgrade_to_head",
        lambda auto_stamp_legacy: migration_flags.append(auto_stamp_legacy),
    )
    monkeypatch.setattr(
        module,
        "uvicorn",
        type(
            "UvicornStub",
            (),
            {
                "run": staticmethod(
                    lambda *args, **kwargs: serve_calls.append((args, kwargs))
                )
            },
        )(),
    )

    module.main()

    assert migration_flags == [True]
    serve_args, serve_kwargs = serve_calls[0]
    assert serve_args == ("ultrasound.api.main:app",)
    assert serve_kwargs == {"host": "0.0.0.0", "port": 8000, "reload": False}


def test_main_skips_migrations_when_requested(monkeypatch) -> None:
    module = importlib.reload(importlib.import_module("ultrasound.api.__main__"))
    migration_called = False
    serve_called = False

    def _mark_migration(_auto_stamp_legacy: bool) -> None:
        nonlocal migration_called
        migration_called = True

    def _mark_server(*_args, **_kwargs) -> None:
        nonlocal serve_called
        serve_called = True

    monkeypatch.setenv("INPHASE_SKIP_MIGRATIONS", "true")
    monkeypatch.setattr(module, "upgrade_to_head", _mark_migration)
    monkeypatch.setattr(
        module,
        "uvicorn",
        type("UvicornStub", (), {"run": staticmethod(_mark_server)})(),
    )

    module.main()

    assert migration_called is False
    assert serve_called is True
