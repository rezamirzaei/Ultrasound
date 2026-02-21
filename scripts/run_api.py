"""Convenience launcher for local API/UI development."""

from __future__ import annotations

import os

import uvicorn

from ultrasound.api.database.migrations import upgrade_to_head

if __name__ == "__main__":
    skip_migrations = os.getenv("INPHASE_SKIP_MIGRATIONS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not skip_migrations:
        auto_stamp_legacy = os.getenv("INPHASE_MIGRATION_AUTO_STAMP", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        upgrade_to_head(auto_stamp_legacy=auto_stamp_legacy)
    uvicorn.run("ultrasound.api.main:app", host="0.0.0.0", port=8000, reload=True)
