"""Coverage for the backward-compatible dataset repository facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ultrasound.api.config import AppConfig
from ultrasound.api.repositories.dataset_repository import DatasetRepository


class _DatabaseStub:
    def __init__(self) -> None:
        self.create_schema_calls = 0

    def create_schema(self) -> None:
        self.create_schema_calls += 1


class _BusiRepositoryStub:
    def __init__(self) -> None:
        self.sync_calls = 0

    def sync_busi_from_filesystem(self) -> int:
        self.sync_calls += 1
        return 11

    def get_busi_counts(self) -> dict[str, int]:
        return {"benign": 2}

    def get_busi_sample(self, *, class_name: str, index: int) -> tuple[str, int]:
        return class_name, index

    def add_busi_uploaded_sample(self, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    def list_busi_training_samples(self, *, include_normal: bool = False) -> list[bool]:
        return [include_normal]

    def save_busi_training_run(self, run: object) -> object:
        return run

    def get_latest_busi_training_run(self, *, include_normal: bool = False) -> tuple[str, bool]:
        return "latest_busi", include_normal


class _IndustrialRepositoryStub:
    def __init__(self) -> None:
        self.sync_calls = 0

    def sync_industrial_from_filesystem(self) -> int:
        self.sync_calls += 1
        return 13

    def add_industrial_uploaded_sample(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    def list_industrial_training_samples(self, dataset_name: str) -> list[str]:
        return [dataset_name]

    def get_industrial_counts(self) -> dict[str, int]:
        return {"steel_defect": 3}

    def get_industrial_annotation_count(self, dataset_name: str) -> int:
        return len(dataset_name)

    def get_industrial_sample(
        self,
        *,
        dataset_name: str,
        split: str,
        class_name: str,
        index: int,
    ) -> tuple[str, str, str, int]:
        return dataset_name, split, class_name, index

    def save_industrial_training_run(self, run: object) -> object:
        return run

    def get_latest_industrial_training_run(self, dataset_name: str) -> tuple[str, str]:
        return "latest_industrial", dataset_name


class _NdtRepositoryStub:
    def __init__(self) -> None:
        self.sync_calls = 0

    def sync_ndt_from_filesystem(self) -> int:
        self.sync_calls += 1
        return 12

    def list_ndt_samples(self) -> list[str]:
        return ["sample_a"]

    def load_ndt_sample(self, sample_name: str) -> dict[str, str]:
        return {"name": sample_name}

    def summarize_ndt_samples(self) -> list[dict[str, int]]:
        return [{"n_points": 5}]


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    busi_dir = data_dir / "busi"
    ui_dir = project_root / "ui"
    artifacts_dir = project_root / "outputs" / "api"
    for path in (ndt_dir, busi_dir, ui_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project_root,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
        database_url="sqlite:///:memory:",
    )


def test_dataset_repository_initialization_creates_schema_and_syncs_legacy_sources(
    tmp_path: Path,
) -> None:
    db = _DatabaseStub()
    busi = _BusiRepositoryStub()
    ndt = _NdtRepositoryStub()
    industrial = _IndustrialRepositoryStub()

    DatasetRepository(
        _make_config(tmp_path),
        cast(Any, db),
        busi_repository=cast(Any, busi),
        ndt_repository=cast(Any, ndt),
        industrial_repository=cast(Any, industrial),
    )

    assert db.create_schema_calls == 1
    assert busi.sync_calls == 1
    assert ndt.sync_calls == 1
    assert industrial.sync_calls == 0


def test_dataset_repository_delegates_all_domain_operations(tmp_path: Path) -> None:
    db = _DatabaseStub()
    busi = _BusiRepositoryStub()
    ndt = _NdtRepositoryStub()
    industrial = _IndustrialRepositoryStub()
    repository = DatasetRepository(
        _make_config(tmp_path),
        cast(Any, db),
        busi_repository=cast(Any, busi),
        ndt_repository=cast(Any, ndt),
        industrial_repository=cast(Any, industrial),
    )

    assert repository.sync_busi_from_filesystem() == 11
    assert repository.sync_ndt_from_filesystem() == 12
    assert repository.sync_industrial_from_filesystem() == 13
    assert repository.get_busi_counts() == {"benign": 2}
    assert repository.get_busi_sample("benign", 3) == ("benign", 3)
    assert repository.add_busi_uploaded_sample("file") == (("file",), {})
    assert repository.list_busi_training_samples(include_normal=True) == [True]
    assert repository.save_busi_training_run({"epochs": 4}) == {"epochs": 4}
    assert repository.get_latest_busi_training_run(include_normal=True) == ("latest_busi", True)
    assert repository.add_industrial_uploaded_sample("file") == (("file",), {})
    assert repository.list_industrial_training_samples("steel_defect") == ["steel_defect"]
    assert repository.get_industrial_counts() == {"steel_defect": 3}
    assert repository.get_industrial_annotation_count("steel_defect") == len("steel_defect")
    assert repository.get_industrial_sample("steel_defect", "train", "crazing", 2) == (
        "steel_defect",
        "train",
        "crazing",
        2,
    )
    assert repository.save_industrial_training_run({"epochs": 2}) == {"epochs": 2}
    assert repository.get_latest_industrial_training_run("steel_defect") == (
        "latest_industrial",
        "steel_defect",
    )
    assert repository.list_ndt_samples() == ["sample_a"]
    assert repository.load_ndt_sample("sample_a") == {"name": "sample_a"}
    assert repository.summarize_ndt_samples() == [{"n_points": 5}]
