"""Dashboard and dataset summary endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.controllers.error_mapping import raise_http_error
from ultrasound.api.models.schemas import (
    BusiSamplePreview,
    BusiTrainingRequest,
    BusiTrainingResponse,
    DashboardSummaryResponse,
    DataReadinessResponse,
    IndustrialDatasetSummaryResponse,
    IndustrialSamplePreview,
    IndustrialSegmentationPreview,
    IndustrialTrainingRequest,
    IndustrialTrainingResponse,
    NdtSampleDetail,
    NdtSampleSummary,
    NdtSignalPreview,
)
from ultrasound.api.services.service_errors import ServiceError

router = APIRouter(
    tags=["dashboard"],
    dependencies=[Depends(require_role("viewer"))],
)


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    container: ApplicationContainer = Depends(get_container),
) -> DashboardSummaryResponse:
    """Return top-level counters consumed by dashboard UI."""
    return container.dashboard_service.get_summary()


@router.get("/dashboard/readiness", response_model=DataReadinessResponse)
def get_dashboard_readiness(
    container: ApplicationContainer = Depends(get_container),
) -> DataReadinessResponse:
    """Return dataset/system readiness diagnostics for operators."""
    return container.dashboard_service.get_data_readiness()


@router.get("/datasets/busi/counts", response_model=dict[str, int])
def get_busi_counts(container: ApplicationContainer = Depends(get_container)) -> dict[str, int]:
    """Return BUSI class counts."""
    return container.dashboard_service.get_busi_counts()


@router.get("/datasets/industrial/summary", response_model=IndustrialDatasetSummaryResponse)
def get_industrial_summary(
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialDatasetSummaryResponse:
    """Return class/split coverage for steel, NEU, and casting datasets."""
    return container.dashboard_service.get_industrial_summary()


@router.get(
    "/datasets/industrial/samples/{dataset_name}/{split}/{class_name}/{sample_index}",
    response_model=IndustrialSamplePreview,
)
def get_industrial_sample_preview(
    dataset_name: str,
    split: str,
    class_name: str,
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialSamplePreview:
    """Return one industrial sample preview for selected dataset/split/class."""
    try:
        return container.dashboard_service.get_industrial_sample_preview(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            sample_index=sample_index,
        )
    except ServiceError as exc:
        raise_http_error(exc)


@router.get("/datasets/industrial/training/latest", response_model=IndustrialTrainingResponse)
def get_latest_industrial_training(
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"] = Query(
        default="steel_defect"
    ),
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialTrainingResponse:
    """Return latest industrial training metrics and learning curve from SQL storage."""
    return container.industrial_training_service.get_latest_run(dataset_name=dataset_name)


@router.post("/datasets/industrial/training/run", response_model=IndustrialTrainingResponse)
def run_industrial_training(
    request: IndustrialTrainingRequest,
    _role: object = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialTrainingResponse:
    """Train industrial classifier from SQL-backed samples and persist run metrics."""
    try:
        return container.industrial_training_service.run_training(request)
    except ServiceError as exc:
        raise_http_error(exc)


@router.get(
    "/datasets/industrial/segmentation/{dataset_name}/{split}/{class_name}/{sample_index}",
    response_model=IndustrialSegmentationPreview,
)
def get_industrial_segmentation_preview(
    dataset_name: str,
    split: str,
    class_name: str,
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialSegmentationPreview:
    """Return one industrial image + segmentation mask preview (annotation-derived when available)."""
    try:
        return container.industrial_training_service.get_segmentation_preview(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            sample_index=sample_index,
        )
    except ServiceError as exc:
        raise_http_error(exc)


@router.get("/datasets/busi/training/latest", response_model=BusiTrainingResponse)
def get_latest_busi_training(
    include_normal: bool = Query(default=False),
    container: ApplicationContainer = Depends(get_container),
) -> BusiTrainingResponse:
    """Return latest BUSI training metrics and learning curve from SQL storage."""
    return container.busi_training_service.get_latest_run(include_normal=include_normal)


@router.post("/datasets/busi/training/run", response_model=BusiTrainingResponse)
def run_busi_training(
    request: BusiTrainingRequest,
    _role: object = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> BusiTrainingResponse:
    """Train a BUSI classifier from SQL-backed samples and persist run metrics."""
    try:
        return container.busi_training_service.run_training(request)
    except ServiceError as exc:
        raise_http_error(exc)


@router.get("/datasets/busi/samples/{class_name}/{sample_index}", response_model=BusiSamplePreview)
def get_busi_sample_preview(
    class_name: Literal["benign", "malignant", "normal"],
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> BusiSamplePreview:
    """Return one BUSI sample and mask preview for UI exploration."""
    try:
        return container.dashboard_service.get_busi_sample_preview(class_name, sample_index)
    except ServiceError as exc:
        raise_http_error(exc)


@router.get("/datasets/ndt/samples", response_model=list[NdtSampleSummary])
def list_ndt_samples(
    container: ApplicationContainer = Depends(get_container),
) -> list[NdtSampleSummary]:
    """Return available NDT samples with metadata summaries."""
    return container.dashboard_service.list_ndt_samples()


@router.get("/datasets/ndt/samples/{sample_name}", response_model=NdtSampleDetail)
def get_ndt_sample(
    sample_name: str,
    container: ApplicationContainer = Depends(get_container),
) -> NdtSampleDetail:
    """Return one NDT sample detail for UI drill-down views."""
    try:
        return container.dashboard_service.get_ndt_sample_detail(sample_name)
    except ServiceError as exc:
        raise_http_error(exc)


@router.get("/datasets/ndt/samples/{sample_name}/signal", response_model=NdtSignalPreview)
def get_ndt_sample_signal(
    sample_name: str,
    max_points: int = Query(default=1024, ge=128, le=4096),
    container: ApplicationContainer = Depends(get_container),
) -> NdtSignalPreview:
    """Return sampled RF waveform data for plotting in UI dashboards."""
    try:
        return container.dashboard_service.get_ndt_signal_preview(
            sample_name, max_points=max_points
        )
    except ServiceError as exc:
        raise_http_error(exc)
