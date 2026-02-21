"""Service layer for dashboard and dataset summary use-cases."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ultrasound.api.models.schemas import (
    BusiSamplePreview,
    DashboardSummaryResponse,
    DataReadinessResponse,
    NdtDefect,
    NdtDefectMarker,
    NdtSampleDetail,
    NdtSampleSummary,
    NdtSignalPreview,
    NdtSignalStats,
    NdtWallMarker,
)
from ultrasound.api.repositories.dataset_repository import DatasetRepository
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.ndt_detection_service import NdtDetectionService


class DashboardService:
    """Business logic for dashboard-level responses."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        media_service: MediaService,
        ndt_detection_service: NdtDetectionService,
    ):
        self.dataset_repository = dataset_repository
        self.media_service = media_service
        self.ndt_detection_service = ndt_detection_service

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
        sample = self.dataset_repository.get_busi_sample(
            class_name=class_name,
            index=sample_index,
        )
        mask_binary = np.asarray(sample.mask > 0, dtype=np.uint8) * 255

        lesion_pixels = int(np.count_nonzero(mask_binary))
        lesion_ratio = float(lesion_pixels / mask_binary.size) if mask_binary.size else 0.0

        return BusiSamplePreview(
            class_name=sample.class_name,
            requested_index=int(sample.requested_index),
            resolved_index=sample.resolved_index,
            total_samples=sample.total_samples,
            image_shape=[int(v) for v in sample.image_rgb.shape],
            lesion_pixels=lesion_pixels,
            lesion_ratio=lesion_ratio,
            image_data_url=self.media_service.as_png_data_url(sample.image_rgb),
            mask_data_url=self.media_service.as_png_data_url(mask_binary),
        )

    def list_ndt_samples(self) -> list[NdtSampleSummary]:
        summaries: list[NdtSampleSummary] = []
        for sample_name in self.dataset_repository.list_ndt_samples():
            sample = self.dataset_repository.load_ndt_sample(sample_name)
            resolved_defects = self.ndt_detection_service.resolve_defects(sample)
            summaries.append(
                NdtSampleSummary(
                    name=sample.name,
                    n_points=int(sample.rf.size),
                    fs_hz=float(sample.fs_hz),
                    fc_hz=float(sample.fc_hz),
                    thickness_mm=(
                        float(sample.thickness_m * 1e3) if sample.thickness_m is not None else None
                    ),
                    n_defects=len(resolved_defects),
                )
            )
        return summaries

    def get_ndt_signal_preview(self, sample_name: str, max_points: int = 1024) -> NdtSignalPreview:
        sample = self.dataset_repository.load_ndt_sample(sample_name)
        signal_analysis = self.ndt_detection_service.analyze_signal(sample)
        resolved_defects = self.ndt_detection_service.resolve_defects(
            sample,
            signal_analysis=signal_analysis,
        )
        rf = np.asarray(sample.rf, dtype=np.float64).reshape(-1)
        time = np.asarray(sample.time, dtype=np.float64).reshape(-1)

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

        c = float(sample.c_mps)
        defect_markers: list[NdtDefectMarker] = []
        for defect in resolved_defects:
            depth_m = defect.depth_m
            if c <= 0.0:
                continue

            if defect.time_us is not None:
                two_way_time_us = float(defect.time_us)
            else:
                two_way_time_us = (2.0 * float(depth_m) / c) * 1e6

            defect_markers.append(
                NdtDefectMarker(
                    depth_mm=float(depth_m * 1e3),
                    amplitude=defect.amplitude,
                    two_way_time_us=two_way_time_us,
                    confidence=defect.confidence,
                    source=defect.source,
                )
            )

        wall_markers: list[NdtWallMarker] = []
        if signal_analysis.front_wall is not None:
            wall_markers.append(
                NdtWallMarker(
                    label="front_wall",
                    depth_mm=0.0,
                    amplitude=signal_analysis.front_wall.amplitude,
                    two_way_time_us=signal_analysis.front_wall.time_us,
                    confidence=signal_analysis.front_wall.confidence,
                    time_std_us=signal_analysis.front_wall.time_std_us,
                )
            )
        if signal_analysis.back_wall is not None:
            wall_markers.append(
                NdtWallMarker(
                    label="back_wall",
                    depth_mm=(
                        signal_analysis.back_wall.depth_m * 1e3
                        if signal_analysis.back_wall.depth_m is not None
                        else None
                    ),
                    amplitude=signal_analysis.back_wall.amplitude,
                    two_way_time_us=signal_analysis.back_wall.time_us,
                    confidence=signal_analysis.back_wall.confidence,
                    time_std_us=signal_analysis.back_wall.time_std_us,
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
            total_peaks=signal_analysis.total_peaks,
            wall_markers=wall_markers,
            estimated_thickness_mm=signal_analysis.estimated_thickness_mm,
            thickness_std_mm=signal_analysis.thickness_std_mm,
            thickness_ci95_lower_mm=signal_analysis.thickness_ci95_lower_mm,
            thickness_ci95_upper_mm=signal_analysis.thickness_ci95_upper_mm,
            thickness_confidence=signal_analysis.thickness_confidence,
            thickness_method=signal_analysis.thickness_method,
            nominal_thickness_mm=signal_analysis.nominal_thickness_mm,
            thickness_error_mm=signal_analysis.thickness_error_mm,
            thinning_flag=signal_analysis.thinning_flag,
            defect_markers=defect_markers,
        )

    def get_ndt_sample_detail(self, sample_name: str) -> NdtSampleDetail:
        sample = self.dataset_repository.load_ndt_sample(sample_name)
        resolved_defects = self.ndt_detection_service.resolve_defects(sample)
        return NdtSampleDetail(
            name=sample.name,
            n_points=int(sample.rf.size),
            fs_hz=float(sample.fs_hz),
            fc_hz=float(sample.fc_hz),
            thickness_mm=(
                float(sample.thickness_m * 1e3) if sample.thickness_m is not None else None
            ),
            n_defects=len(resolved_defects),
            description=sample.description,
            defects=[
                NdtDefect(
                    depth_m=defect.depth_m,
                    amplitude=defect.amplitude,
                    time_us=defect.time_us,
                    confidence=defect.confidence,
                    source=defect.source,
                )
                for defect in resolved_defects
            ],
        )
