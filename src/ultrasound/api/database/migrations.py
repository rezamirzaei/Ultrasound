"""Alembic migration helpers for runtime startup workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alembic.config import Config

logger = logging.getLogger("inphase.migrations")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _alembic_config(database_url: str | None = None) -> Config:
    from alembic.config import Config

    root = _project_root()
    cfg = Config(str(root / "alembic.ini"))
    resolved_url = database_url or os.getenv("INPHASE_DATABASE_URL")
    if resolved_url:
        cfg.set_main_option("sqlalchemy.url", resolved_url)
    return cfg


def _legacy_stamp_revision(config: Config) -> str:
    """Best-effort baseline revision for legacy pre-Alembic databases."""
    try:
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(config)
        bases = tuple(script.get_bases())
        if bases:
            return str(bases[0])
    except Exception:
        logger.exception("Could not resolve baseline Alembic revision for legacy auto-stamp.")
    return "head"


def upgrade_to_head(database_url: str | None = None, auto_stamp_legacy: bool = True) -> None:
    """Upgrade schema to latest revision; optionally stamp legacy pre-Alembic DBs."""
    try:
        from alembic import command
    except ModuleNotFoundError:
        logger.warning(
            "Alembic is not installed in this runtime image. "
            "Skipping migration step and continuing startup."
        )
        return

    config = _alembic_config(database_url=database_url)

    try:
        command.upgrade(config, "head")
    except Exception as exc:
        message = str(exc).lower()
        is_legacy_conflict = "already exists" in message and "table" in message
        if auto_stamp_legacy and is_legacy_conflict:
            legacy_revision = _legacy_stamp_revision(config)
            logger.warning(
                "Detected pre-Alembic schema, stamping DB to revision %s before upgrading to head. "
                "error=%s",
                legacy_revision,
                exc,
            )
            command.stamp(config, legacy_revision)
            if legacy_revision != "head":
                command.upgrade(config, "head")
            return
        raise
