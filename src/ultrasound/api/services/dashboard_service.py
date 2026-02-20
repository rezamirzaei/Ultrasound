"""Service layer for dashboard and dataset summary use-cases."""

from __future__ import annotations

from datetime import datetime, timezone

from ultrasound.api.models.schemas import (
    DashboardSummaryResponse,
    NdtSampleDetail,
    NdtSampleSummary,
)
from ultrasound.api.repositories.dataset_repository import DatasetRepository


class DashboardService:
    """Business logic for dashboard-level responses."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def get_summary(self) -> DashboardSummaryResponse:
        busi_counts = self.dataset_repository.get_busi_counts()
        ndt_samples = self.dataset_repository.list_ndt_samples()
        return DashboardSummaryResponse(
            busi_counts=busi_counts,
            busi_total=int(sum(busi_counts.values())),
            ndt_samples=len(ndt_samples),
            generated_at=datetime.now(tz=timezone.utc),
        )

    def get_busi_counts(self) -> dict[str, int]:
        return self.dataset_repository.get_busi_counts()

    def list_ndt_samples(self) -> list[NdtSampleSummary]:
        rows = self.dataset_repository.summarize_ndt_samples()
        return [
            NdtSampleSummary(
                name=row["name"],
                n_points=row["n_points"],
                fs_hz=row["fs_hz"],
                fc_hz=row["fc_hz"],
                thickness_mm=row["thickness_mm"],
                n_defects=row["n_defects"],
            )
            for row in rows
        ]

    def get_ndt_sample_detail(self, sample_name: str) -> NdtSampleDetail:
        sample = self.dataset_repository.load_ndt_sample(sample_name)
        return NdtSampleDetail(
            name=sample["name"],
            n_points=int(sample["rf"].size),
            fs_hz=float(sample["fs"]),
            fc_hz=float(sample["fc"]),
            thickness_mm=float(sample["thickness"] * 1e3),
            n_defects=len(sample["defects"]),
            description=sample["description"],
            defects=sample["defects"],
        )
