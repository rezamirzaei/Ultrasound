"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal

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
    status: str = Field(pattern="^(ok|warning)$")
    busi_available_classes: List[str]
    busi_missing_classes: List[str]
    ndt_samples: int
    issues: List[str]
    generated_at: datetime


class NdtSampleSummary(BaseModel):
    name: str
    n_points: int = Field(gt=0)
    fs_hz: float = Field(gt=0)
    fc_hz: float = Field(gt=0)
    thickness_mm: float | None = None
    n_defects: int = Field(ge=0)


class NdtDefect(BaseModel):
    depth_m: float | None = None
    amplitude: float | None = None
    time_us: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str | None = Field(default=None, pattern="^(metadata|signal|fused)$")


class NdtSampleDetail(NdtSampleSummary):
    description: str
    defects: List[NdtDefect]


class NdtSignalStats(BaseModel):
    amplitude_min: float
    amplitude_max: float
    amplitude_rms: float = Field(ge=0)
    time_start_us: float
    time_end_us: float


class NdtDefectMarker(BaseModel):
    depth_mm: float = Field(ge=0)
    amplitude: float | None = None
    two_way_time_us: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str | None = Field(default=None, pattern="^(metadata|signal|fused)$")


class NdtWallMarker(BaseModel):
    label: str = Field(pattern="^(front_wall|back_wall)$")
    depth_mm: float | None = Field(default=None, ge=0)
    amplitude: float | None = None
    two_way_time_us: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    time_std_us: float | None = Field(default=None, ge=0)


class NdtSignalPreview(BaseModel):
    sample_name: str
    n_original_points: int = Field(gt=0)
    n_sampled_points: int = Field(gt=0)
    time_us: List[float]
    rf: List[float]
    stats: NdtSignalStats
    total_peaks: int = Field(ge=0)
    wall_markers: List[NdtWallMarker] = Field(default_factory=list)
    estimated_thickness_mm: float | None = Field(default=None, ge=0)
    thickness_std_mm: float | None = Field(default=None, ge=0)
    thickness_ci95_lower_mm: float | None = Field(default=None, ge=0)
    thickness_ci95_upper_mm: float | None = Field(default=None, ge=0)
    thickness_confidence: float | None = Field(default=None, ge=0, le=1)
    thickness_method: Literal["time_of_flight", "absolute_backwall", "insufficient_data"] = (
        "insufficient_data"
    )
    nominal_thickness_mm: float | None = Field(default=None, ge=0)
    thickness_error_mm: float | None = None
    thinning_flag: bool = False
    defect_markers: List[NdtDefectMarker]


class BusiSamplePreview(BaseModel):
    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    image_shape: List[int]
    lesion_pixels: int = Field(ge=0)
    lesion_ratio: float = Field(ge=0.0, le=1.0)
    image_data_url: str
    mask_data_url: str


class BusiTrainingRequest(BaseModel):
    include_normal: bool = False
    epochs: int = Field(default=12, ge=2, le=100)
    batch_size: int = Field(default=16, ge=4, le=256)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)


class BusiTrainingCurvePoint(BaseModel):
    epoch: int = Field(ge=1)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)


class BusiTrainingResponse(BaseModel):
    run_id: int | None = Field(default=None, ge=1)
    generated_at: datetime
    storage: Literal["sql"] = "sql"
    include_normal: bool = False
    epochs: int = Field(ge=0)
    batch_size: int = Field(ge=0)
    learning_rate: float = Field(ge=0.0)
    train_samples: int = Field(ge=0)
    test_samples: int = Field(ge=0)
    class_counts: Dict[str, int]
    class_labels: List[str]
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    test_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    train_loss: float | None = Field(default=None, ge=0.0)
    test_loss: float | None = Field(default=None, ge=0.0)
    curve: List[BusiTrainingCurvePoint] = Field(default_factory=list)
    notes: str | None = None


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
    request_id: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    code: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=4, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    username: str
    role: Literal["viewer", "analyst", "admin"]
    expires_at: datetime


class AuthMeResponse(BaseModel):
    username: str
    role: Literal["viewer", "analyst", "admin"]
    expires_at: datetime


class LogoutResponse(BaseModel):
    success: bool
    username: str
    revoked_token: bool


class OpsErrorEvent(BaseModel):
    occurred_at: datetime
    request_id: str
    method: str
    path: str
    status_code: int = Field(ge=100, le=599)
    detail: str
    role: Literal["viewer", "analyst", "admin"] | None = None


class OpsErrorSummaryResponse(BaseModel):
    generated_at: datetime
    window_minutes: int = Field(ge=1)
    total_error_count: int = Field(ge=0)
    recent_error_count: int = Field(ge=0)
    by_status: Dict[str, int]
    by_path: Dict[str, int]
    last_error_at: datetime | None = None


class DatasetResyncResponse(BaseModel):
    generated_at: datetime
    busi_rows_synced: int = Field(ge=0)
    ndt_rows_synced: int = Field(ge=0)
