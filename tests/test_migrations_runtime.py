"""Tests for runtime Alembic migration behavior."""

from __future__ import annotations

import importlib
import sys
import types


def test_upgrade_to_head_stamps_baseline_then_retries(monkeypatch) -> None:
    module = importlib.reload(importlib.import_module("ultrasound.api.database.migrations"))
    config = object()
    calls: list[tuple[str, str]] = []

    def _upgrade(cfg: object, revision: str) -> None:
        assert cfg is config
        calls.append(("upgrade", revision))
        if len(calls) == 1:
            raise RuntimeError("table auth_users already exists")

    def _stamp(cfg: object, revision: str) -> None:
        assert cfg is config
        calls.append(("stamp", revision))

    fake_alembic = types.ModuleType("alembic")
    fake_alembic.command = types.SimpleNamespace(upgrade=_upgrade, stamp=_stamp)

    monkeypatch.setitem(sys.modules, "alembic", fake_alembic)
    monkeypatch.setattr(module, "_alembic_config", lambda database_url=None: config)
    monkeypatch.setattr(module, "_legacy_stamp_revision", lambda _config: "20260221_0001")

    module.upgrade_to_head(database_url="sqlite:///tmp.db", auto_stamp_legacy=True)

    assert calls == [
        ("upgrade", "head"),
        ("stamp", "20260221_0001"),
        ("upgrade", "head"),
    ]


def test_upgrade_to_head_re_raises_without_auto_stamp(monkeypatch) -> None:
    module = importlib.reload(importlib.import_module("ultrasound.api.database.migrations"))
    config = object()

    def _upgrade(cfg: object, revision: str) -> None:
        assert cfg is config
        raise RuntimeError("table auth_users already exists")

    fake_alembic = types.ModuleType("alembic")
    fake_alembic.command = types.SimpleNamespace(upgrade=_upgrade, stamp=lambda *_args: None)

    monkeypatch.setitem(sys.modules, "alembic", fake_alembic)
    monkeypatch.setattr(module, "_alembic_config", lambda database_url=None: config)

    try:
        module.upgrade_to_head(auto_stamp_legacy=False)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError")
