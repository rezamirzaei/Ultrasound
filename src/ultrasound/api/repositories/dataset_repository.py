"""Compatibility facade bundling domain-specific repositories."""

from __future__ import annotations

from typing import Any

from ultrasound.api.config import AppConfig
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.repositories.busi_repository import BusiRepository
from ultrasound.api.repositories.industrial_repository import IndustrialRepository
from ultrasound.api.repositories.ndt_repository import NdtRepository


class DatasetRepository:
    """Backward-compatible facade over split BUSI, NDT, and industrial repositories."""

    def __init__(
        self,
        config: AppConfig,
        db: DatabaseSessionManager,
        *,
        busi_repository: BusiRepository | None = None,
        ndt_repository: NdtRepository | None = None,
        industrial_repository: IndustrialRepository | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.db.create_schema()
        self.busi = busi_repository or BusiRepository(config, db)
        self.ndt = ndt_repository or NdtRepository(config, db)
        self.industrial = industrial_repository or IndustrialRepository(config, db)
        self.sync_from_sources()

    def sync_from_sources(self) -> None:
        self.busi.sync_busi_from_filesystem()
        self.ndt.sync_ndt_from_filesystem()

    def sync_busi_from_filesystem(self) -> int:
        return self.busi.sync_busi_from_filesystem()

    def sync_ndt_from_filesystem(self) -> int:
        return self.ndt.sync_ndt_from_filesystem()

    def sync_industrial_from_filesystem(self) -> int:
        return self.industrial.sync_industrial_from_filesystem()

    def get_busi_counts(self) -> dict[str, int]:
        return self.busi.get_busi_counts()

    def get_busi_sample(self, class_name: str, index: int = 0):
        return self.busi.get_busi_sample(class_name=class_name, index=index)

    def add_busi_uploaded_sample(self, *args: Any, **kwargs: Any):
        return self.busi.add_busi_uploaded_sample(*args, **kwargs)

    def list_busi_training_samples(self, include_normal: bool = False):
        return self.busi.list_busi_training_samples(include_normal=include_normal)

    def save_busi_training_run(self, run):
        return self.busi.save_busi_training_run(run)

    def get_latest_busi_training_run(self, include_normal: bool = False):
        return self.busi.get_latest_busi_training_run(include_normal=include_normal)

    def add_industrial_uploaded_sample(self, *args: Any, **kwargs: Any):
        return self.industrial.add_industrial_uploaded_sample(*args, **kwargs)

    def list_industrial_training_samples(self, dataset_name: str):
        return self.industrial.list_industrial_training_samples(dataset_name)

    def get_industrial_counts(self):
        return self.industrial.get_industrial_counts()

    def get_industrial_annotation_count(self, dataset_name: str) -> int:
        return self.industrial.get_industrial_annotation_count(dataset_name)

    def get_industrial_sample(self, dataset_name: str, split: str, class_name: str, index: int = 0):
        return self.industrial.get_industrial_sample(dataset_name=dataset_name, split=split, class_name=class_name, index=index)

    def save_industrial_training_run(self, run):
        return self.industrial.save_industrial_training_run(run)

    def get_latest_industrial_training_run(self, dataset_name: str):
        return self.industrial.get_latest_industrial_training_run(dataset_name)

    def list_ndt_samples(self) -> list[str]:
        return self.ndt.list_ndt_samples()

    def load_ndt_sample(self, sample_name: str):
        return self.ndt.load_ndt_sample(sample_name)

    def summarize_ndt_samples(self):
        return self.ndt.summarize_ndt_samples()
