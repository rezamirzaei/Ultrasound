"""Dashboard and dataset summary endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container
from ultrasound.api.models.schemas import (
    BusiSamplePreview,
    DashboardSummaryResponse,
    NdtSampleDetail,
    NdtSampleSummary,
)

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    container: ApplicationContainer = Depends(get_container),
) -> DashboardSummaryResponse:
    """Return top-level counters consumed by dashboard UI."""
    return container.dashboard_service.get_summary()


@router.get("/datasets/busi/counts", response_model=dict[str, int])
def get_busi_counts(container: ApplicationContainer = Depends(get_container)) -> dict[str, int]:
    """Return BUSI class counts."""
    return container.dashboard_service.get_busi_counts()


@router.get("/datasets/busi/samples/{class_name}/{sample_index}", response_model=BusiSamplePreview)
def get_busi_sample_preview(
    class_name: Literal["benign", "malignant", "normal"],
    sample_index: int,
    container: ApplicationContainer = Depends(get_container),
) -> BusiSamplePreview:
    """Return one BUSI sample and mask preview for UI exploration."""
    try:
        return container.dashboard_service.get_busi_sample_preview(class_name, sample_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
