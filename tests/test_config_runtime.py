"""Runtime tests for AppConfig path and env resolution."""

from __future__ import annotations

from pathlib import Path

from ultrasound.api.config import AppConfig


def test_from_project_root_accepts_file_start_and_blank_database_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    nested_dir = project_root / "nested" / "leaf"
    start_file = nested_dir / "script.py"
    project_root.mkdir(parents=True, exist_ok=True)
    nested_dir.mkdir(parents=True, exist_ok=True)
    start_file.write_text("print('x')\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    (project_root / "src").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("INPHASE_DATABASE_URL", "   ")

    config = AppConfig.from_project_root(start=start_file)

    assert config.project_root == project_root
    assert config.data_dir == project_root / "data"
    assert config.database_url == f"sqlite:///{(project_root / 'data' / 'inphase.sqlite3').resolve()}"


def test_from_project_root_prefers_explicit_database_env(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INPHASE_DATABASE_URL", "sqlite:///custom.sqlite3")

    config = AppConfig.from_project_root(start=project_root)

    assert config.database_url == "sqlite:///custom.sqlite3"
