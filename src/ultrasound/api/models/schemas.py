"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class DashboardSummaryResponse(BaseModel):
    busi_counts: dict[str, int]
    busi_total: int
    ndt_samples: int
    industrial_total: int = Field(default=0, ge=0)
    industrial_datasets: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime


class DataReadinessResponse(BaseModel):
    status: str = Field(pattern="^(ok|warning)$")
    busi_available_classes: list[str]
    busi_missing_classes: list[str]
    ndt_samples: int
    industrial_datasets: dict[str, int] = Field(default_factory=dict)
    issues: list[str]
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
    defects: list[NdtDefect]


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
    time_us: list[float]
    rf: list[float]
    stats: NdtSignalStats
    total_peaks: int = Field(ge=0)
    wall_markers: list[NdtWallMarker] = Field(default_factory=list)
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
    defect_markers: list[NdtDefectMarker]


class BusiSamplePreview(BaseModel):
    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    image_shape: list[int]
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
    class_counts: dict[str, int]
    class_labels: list[str]
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    test_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    train_loss: float | None = Field(default=None, ge=0.0)
    test_loss: float | None = Field(default=None, ge=0.0)
    curve: list[BusiTrainingCurvePoint] = Field(default_factory=list)
    notes: str | None = None


class IndustrialTrainingRequest(BaseModel):
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    epochs: int = Field(default=12, ge=2, le=100)
    batch_size: int = Field(default=16, ge=4, le=256)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)


class IndustrialTrainingCurvePoint(BaseModel):
    epoch: int = Field(ge=1)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)


class IndustrialTrainingResponse(BaseModel):
    run_id: int | None = Field(default=None, ge=1)
    generated_at: datetime
    storage: Literal["sql"] = "sql"
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    epochs: int = Field(ge=0)
    batch_size: int = Field(ge=0)
    learning_rate: float = Field(ge=0.0)
    train_samples: int = Field(ge=0)
    test_samples: int = Field(ge=0)
    class_counts: dict[str, int]
    class_labels: list[str]
    task_type: Literal[
        "classification_single_label",
        "classification_single_label_with_bbox",
    ] = "classification_single_label"
    classification_mode: Literal["binary", "multiclass"] = "multiclass"
    label_source: Literal["folder_name", "folder_name_plus_xml_bbox"] = "folder_name"
    segmentation_supported: bool = False
    segmentation_notes: str | None = None
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    test_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    train_loss: float | None = Field(default=None, ge=0.0)
    test_loss: float | None = Field(default=None, ge=0.0)
    curve: list[IndustrialTrainingCurvePoint] = Field(default_factory=list)
    annotated_samples: int = Field(default=0, ge=0)
    segmentation_iou_train: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_iou_test: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_dice_train: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_dice_test: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class JobEnqueueResponse(BaseModel):
    job_id: int = Field(ge=1)
    job_type: Literal["busi_training", "dataset_resync", "industrial_training"]
    status: Literal["pending", "running", "completed", "failed"]
    requested_by: str
    submitted_at: datetime


class JobRunResponse(JobEnqueueResponse):
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class BusiUploadResponse(BaseModel):
    sample_id: int = Field(ge=1)
    class_name: Literal["benign", "malignant", "normal"]
    split: Literal["train", "test"]
    image_filename: str
    total_class_samples: int = Field(ge=1)
    storage: Literal["sql"] = "sql"
    created_at: datetime


class IndustrialUploadResponse(BaseModel):
    sample_id: int = Field(ge=1)
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    split: str
    class_name: str
    image_filename: str
    relative_path: str
    has_annotation: bool = False
    total_class_samples: int = Field(ge=1)
    storage: Literal["sql"] = "sql"
    created_at: datetime


class IndustrialDatasetRow(BaseModel):
    dataset_name: str
    split: str
    class_name: str
    sample_count: int = Field(ge=0)


class IndustrialDatasetSummaryResponse(BaseModel):
    generated_at: datetime
    total_samples: int = Field(ge=0)
    totals_by_dataset: dict[str, int] = Field(default_factory=dict)
    rows: list[IndustrialDatasetRow] = Field(default_factory=list)


class IndustrialSamplePreview(BaseModel):
    dataset_name: str
    split: str
    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    image_shape: list[int]
    has_annotation: bool = False
    image_data_url: str
    relative_path: str


class IndustrialSegmentationPreview(BaseModel):
    dataset_name: str
    split: str
    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    image_shape: list[int]
    bbox_count: int = Field(default=0, ge=0)
    annotation_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    task_type: Literal[
        "classification_single_label",
        "classification_single_label_with_bbox",
    ] = "classification_single_label"
    segmentation_supported: bool = False
    message: str | None = None
    source: Literal["annotation_xml", "none"] = "none"
    image_data_url: str
    mask_data_url: str
    relative_path: str


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
    image_shape: list[int]
    original_image_data_url: str
    methods: list[MethodPreview]
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
    by_status: dict[str, int]
    by_path: dict[str, int]
    last_error_at: datetime | None = None


class DatasetResyncResponse(BaseModel):
    generated_at: datetime
    busi_rows_synced: int = Field(ge=0)
    ndt_rows_synced: int = Field(ge=0)
    industrial_rows_synced: int = Field(default=0, ge=0)


class DatabaseTableStatus(BaseModel):
    table_name: str
    row_count: int = Field(ge=0)


class DatabaseSchemaStatusResponse(BaseModel):
    generated_at: datetime
    database_url: str
    alembic_current_revision: str | None = None
    alembic_head_revision: str | None = None
    tables: list[DatabaseTableStatus] = Field(default_factory=list)


class YoloStatusResponse(BaseModel):
    available: bool
    backend: str
    default_models: list[str] = Field(default_factory=list)


class BusiYoloModelStatus(BaseModel):
    """Status for the recommended BUSI ultrasound YOLO weights used by the UI lab."""

    model_id: str
    source_url: str
    local_path: str
    downloaded: bool
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    downloaded_at: datetime | None = None


class BusiYoloLabStatusResponse(BaseModel):
    generated_at: datetime
    yolo: YoloStatusResponse
    model: BusiYoloModelStatus
    yolo_class_names: list[str] = Field(default_factory=list)


class YoloLabel(BaseModel):
    """One YOLO-format label row (normalized xywh)."""

    class_id: int = Field(ge=0)
    class_name: str | None = None
    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class YoloPredictRequest(BaseModel):
    model: str = Field(default="yolov8n.pt", min_length=1, max_length=256)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=160, le=2048)
    max_detections: int = Field(default=100, ge=1, le=1000)


class YoloXyxyBox(BaseModel):
    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float = Field(ge=0.0)
    y2: float = Field(ge=0.0)


class BusiYoloSampleResponse(BaseModel):
    """BUSI sample preview augmented with derived YOLO labels from the segmentation mask."""

    sample: BusiSamplePreview
    yolo_class_names: list[str] = Field(default_factory=list)
    yolo_labels: list[YoloLabel] = Field(default_factory=list)
    raw_yolo_labels: str | None = None
    bbox_xyxy: YoloXyxyBox | None = None


class YoloDetection(BaseModel):
    class_id: int
    class_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: YoloXyxyBox


class YoloPredictResponse(BaseModel):
    generated_at: datetime
    model: str
    image_shape: list[int]
    detections: list[YoloDetection] = Field(default_factory=list)
    annotated_image_data_url: str
    backend: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# Liver Ultrasound Detection Lab
# ---------------------------------------------------------------------------

class LiverDatasetStatusResponse(BaseModel):
    """Status of the liver ultrasound detection dataset on disk."""

    ready: bool
    summary: dict[str, Any]
    generated_at: datetime


class LiverYoloLabStatusResponse(BaseModel):
    """Combined status for the liver YOLO detection lab."""

    generated_at: datetime
    yolo: YoloStatusResponse
    dataset: LiverDatasetStatusResponse
    class_names: list[str] = Field(default_factory=list)
    trained_weights: str | None = None
    default_model: str = "yolo11n.pt"


class LiverSampleBbox(BaseModel):
    """One bounding box from the liver CSV annotations."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int = Field(ge=0)
    class_name: str | None = None


class LiverSampleResponse(BaseModel):
    """Preview of a liver ultrasound sample with bbox annotations."""

    category: str
    sample_index: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    image_id: str
    image_shape: list[int]
    bboxes: list[LiverSampleBbox] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    image_data_url: str


# ---------------------------------------------------------------------------
# YOLO Training
# ---------------------------------------------------------------------------

class YoloTrainRequest(BaseModel):
    """Request body for launching a YOLO training run."""

    pretrained_weights: str = Field(default="yolo11n.pt", min_length=1)
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=128)
    image_size: int = Field(default=640, ge=160, le=2048)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    patience: int = Field(default=10, ge=1, le=100)
    train_ratio: float = Field(default=0.8, gt=0.1, lt=1.0)
    freeze_layers: int = Field(default=0, ge=0, le=50)
    use_synthetic: bool = Field(default=False)
    synthetic_samples: int = Field(default=30, ge=10, le=500)


class YoloTrainResponse(BaseModel):
    """Result of a YOLO training run."""

    generated_at: datetime
    best_weights: str | None = None
    last_weights: str | None = None
    epochs_completed: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    run_dir: str | None = None


# ---------------------------------------------------------------------------
# Downloaded asset provenance
# ---------------------------------------------------------------------------

class DownloadedAssetManifest(BaseModel):
    """Minimal provenance for downloaded YOLO assets (weights, sample images)."""

    source_url: str = Field(min_length=8, max_length=2048)
    downloaded_at: datetime
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)

