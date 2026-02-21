"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class DashboardSummaryResponse(BaseModel):
    busi_counts: Dict[str, int]
    busi_total: int
    ndt_samples: int
    generated_at: datetime


class DataReadinessResponse(BaseModel):
    status: str
    busi_available_classes: List[str]
    busi_missing_classes: List[str]
    ndt_samples: int
    issues: List[str]
    generated_at: datetime


class NdtSampleSummary(BaseModel):
    name: str
    n_points: int
    fs_hz: float
    fc_hz: float
    thickness_mm: float
    n_defects: int


class NdtSampleDetail(NdtSampleSummary):
    description: str
    defects: List[dict]


class NdtSignalStats(BaseModel):
    amplitude_min: float
    amplitude_max: float
    amplitude_rms: float
    time_start_us: float
    time_end_us: float


class NdtDefectMarker(BaseModel):
    depth_mm: float
    amplitude: float
    two_way_time_us: float


class NdtSignalPreview(BaseModel):
    sample_name: str
    n_original_points: int
    n_sampled_points: int
    time_us: List[float]
    rf: List[float]
    stats: NdtSignalStats
    defect_markers: List[NdtDefectMarker]


class BusiSamplePreview(BaseModel):
    class_name: str
    requested_index: int
    resolved_index: int
    total_samples: int
    image_shape: List[int]
    lesion_pixels: int
    lesion_ratio: float
    image_data_url: str
    mask_data_url: str


class PreprocessingRequest(BaseModel):
    class_name: str = Field(default="benign", pattern="^(benign|malignant|normal)$")
    sample_index: int = Field(default=0, ge=0)
    lambda_tv: float = Field(default=0.06, gt=0)
    rho: float = Field(default=1.0, gt=0)
    n_iter: int = Field(default=35, ge=5, le=200)
    clip_limit: float = Field(default=2.5, gt=0.1, le=20.0)


class MethodMetrics(BaseModel):
    rmse: float
    psnr: float
    ssim: float
    cv: float


class MethodPreview(BaseModel):
    name: str
    image_data_url: str
    metrics: MethodMetrics


class PreprocessingPreviewResponse(BaseModel):
    image_shape: List[int]
    original_image_data_url: str
    methods: List[MethodPreview]
    recommendation: str
    generated_at: datetime


class ApiError(BaseModel):
    detail: str
