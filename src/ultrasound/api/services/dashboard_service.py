"""Service layer for dashboard and dataset summary use-cases."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from PIL import Image

from ultrasound.api.models.schemas import (
    BusiSamplePreview,
    DashboardSummaryResponse,
    DataReadinessResponse,
    NdtDefectMarker,
    NdtSampleDetail,
    NdtSampleSummary,
    NdtSignalPreview,
    NdtSignalStats,
)
from ultrasound.api.repositories.dataset_repository import DatasetRepository


class DashboardService:
    """Business logic for dashboard-level responses."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def _as_data_url(self, image: np.ndarray) -> str:
        """Convert grayscale/RGB arrays to PNG data URLs consumable by the UI."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        else:
            pil = Image.fromarray(image)

        buffer = BytesIO()
        pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def get_summary(self) -> DashboardSummaryResponse:
        busi_counts = self.dataset_repository.get_busi_counts()
        ndt_samples = self.dataset_repository.list_ndt_samples()
        return DashboardSummaryResponse(
            busi_counts=busi_counts,
            busi_total=int(sum(busi_counts.values())),
            ndt_samples=len(ndt_samples),
            generated_at=datetime.now(tz=timezone.utc),
        )

    def get_data_readiness(self) -> DataReadinessResponse:
        busi_counts = self.dataset_repository.get_busi_counts()
        ndt_samples = self.dataset_repository.list_ndt_samples()

        busi_available_classes = [name for name, count in busi_counts.items() if count > 0]
        busi_missing_classes = [name for name, count in busi_counts.items() if count <= 0]

        issues: list[str] = []
        if busi_missing_classes:
            issues.append(
                "BUSI classes with no samples: " + ", ".join(sorted(busi_missing_classes))
            )
        if len(ndt_samples) == 0:
            issues.append("No NDT samples were found in data/ascan_signals/ndt_samples.")

        status = "ok" if not issues else "warning"
        return DataReadinessResponse(
            status=status,
            busi_available_classes=sorted(busi_available_classes),
            busi_missing_classes=sorted(busi_missing_classes),
            ndt_samples=len(ndt_samples),
            issues=issues,
            generated_at=datetime.now(tz=timezone.utc),
        )

    def get_busi_counts(self) -> dict[str, int]:
        return self.dataset_repository.get_busi_counts()

    def get_busi_sample_preview(self, class_name: str, sample_index: int) -> BusiSamplePreview:
        counts = self.dataset_repository.get_busi_counts()
        total = int(counts.get(class_name, 0))
        if total <= 0:
            raise FileNotFoundError(f"No BUSI samples available for class '{class_name}'.")

        resolved_index = int(sample_index % total)
        image_rgb, mask = self.dataset_repository.get_busi_sample(
            class_name=class_name,
            index=resolved_index,
        )
        mask_binary = np.asarray(mask > 0, dtype=np.uint8) * 255

        lesion_pixels = int(np.count_nonzero(mask_binary))
        lesion_ratio = float(lesion_pixels / mask_binary.size) if mask_binary.size else 0.0

        return BusiSamplePreview(
            class_name=class_name,
            requested_index=int(sample_index),
            resolved_index=resolved_index,
            total_samples=total,
            image_shape=[int(v) for v in image_rgb.shape],
            lesion_pixels=lesion_pixels,
            lesion_ratio=lesion_ratio,
            image_data_url=self._as_data_url(image_rgb),
            mask_data_url=self._as_data_url(mask_binary),
        )

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

    def get_ndt_signal_preview(self, sample_name: str, max_points: int = 1024) -> NdtSignalPreview:
        sample = self.dataset_repository.load_ndt_sample(sample_name)
        rf = np.asarray(sample["rf"], dtype=np.float64).reshape(-1)
        time = np.asarray(sample["time"], dtype=np.float64).reshape(-1)

        n_original = int(rf.size)
        if n_original == 0:
            raise ValueError(f"NDT sample '{sample_name}' contains no RF points.")

        n_target = max(1, min(int(max_points), n_original))
        if n_target < n_original:
            idx = np.linspace(0, n_original - 1, num=n_target, dtype=np.int64)
            rf_sampled = rf[idx]
            time_sampled = time[idx]
        else:
            rf_sampled = rf
            time_sampled = time

        c = float(sample["c"])
        defect_markers: list[NdtDefectMarker] = []
        for defect in sample["defects"]:
            depth_m = float(defect.get("depth_m", np.nan))
            amplitude = float(defect.get("amplitude", np.nan))
            if not np.isfinite(depth_m) or c <= 0:
                continue
            two_way_time_us = (2.0 * depth_m / c) * 1e6
            defect_markers.append(
                NdtDefectMarker(
                    depth_mm=depth_m * 1e3,
                    amplitude=amplitude,
                    two_way_time_us=two_way_time_us,
                )
            )

        stats = NdtSignalStats(
            amplitude_min=float(np.min(rf_sampled)),
            amplitude_max=float(np.max(rf_sampled)),
            amplitude_rms=float(np.sqrt(np.mean(np.square(rf_sampled)))),
            time_start_us=float(time_sampled[0] * 1e6),
            time_end_us=float(time_sampled[-1] * 1e6),
        )

        return NdtSignalPreview(
            sample_name=sample_name,
            n_original_points=n_original,
            n_sampled_points=int(rf_sampled.size),
            time_us=[float(v * 1e6) for v in time_sampled],
            rf=[float(v) for v in rf_sampled],
            stats=stats,
            defect_markers=defect_markers,
        )

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
