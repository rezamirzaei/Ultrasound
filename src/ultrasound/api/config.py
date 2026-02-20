"""Configuration objects for the web API layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for API/UI paths and project resources."""

    project_root: Path
    data_dir: Path
    busi_dir: Path
    ndt_dir: Path
    ui_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_project_root(cls, start: Path | None = None) -> "AppConfig":
        """Resolve project root and construct default path configuration."""
        cursor = (start or Path.cwd()).resolve()
        for candidate in (cursor, *cursor.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                project_root = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not locate project root containing pyproject.toml and src/"
            )

        data_dir = project_root / "data"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            busi_dir=data_dir / "busi",
            ndt_dir=data_dir / "ascan_signals" / "ndt_samples",
            ui_dir=project_root / "ui",
            artifacts_dir=project_root / "outputs" / "api",
        )
