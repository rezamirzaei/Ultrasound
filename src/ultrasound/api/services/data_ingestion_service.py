"""Data ingestion and synchronization services."""

from __future__ import annotations

from datetime import datetime, timezone

from ultrasound.api.models.schemas import DatasetResyncResponse
from ultrasound.api.repositories.dataset_repository import DatasetRepository


class DataIngestionService:
    """Manages controlled sync from source files into database tables."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def resync_all(self) -> DatasetResyncResponse:
        busi_rows = self.dataset_repository.sync_busi_from_filesystem()
        ndt_rows = self.dataset_repository.sync_ndt_from_filesystem()
        return DatasetResyncResponse(
            generated_at=datetime.now(tz=timezone.utc),
            busi_rows_synced=int(busi_rows),
            ndt_rows_synced=int(ndt_rows),
        )
