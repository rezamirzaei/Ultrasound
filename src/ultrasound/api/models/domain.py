"""Validated domain objects used across repositories and services."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _to_finite_float_or_none(value: object) -> float | None:
    """Parse scalar-like values into finite floats, otherwise return None."""
    if value is None:
        return None
    try:
        parsed = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


class NdtDefectRecord(BaseModel):
    """Normalized one-defect record extracted from NDT metadata."""

    depth_m: float | None = None
    amplitude: float | None = None

    @field_validator("depth_m", "amplitude", mode="before")
    @classmethod
    def parse_optional_floats(cls, value: object) -> float | None:
        return _to_finite_float_or_none(value)


class NdtAnalyzedDefect(BaseModel):
    """Defect candidate produced by metadata and/or waveform analysis."""

    depth_m: float = Field(ge=0.0)
    amplitude: float | None = None
    time_us: float | None = Field(default=None, ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["metadata", "signal", "fused"]

    @field_validator("amplitude", mode="before")
    @classmethod
    def parse_optional_amplitude(cls, value: object) -> float | None:
        return _to_finite_float_or_none(value)

    @field_validator("time_us", mode="before")
    @classmethod
    def parse_optional_time(cls, value: object) -> float | None:
        return _to_finite_float_or_none(value)


class NdtWallEchoRecord(BaseModel):
    """Detected front/back wall echo descriptor."""

    label: Literal["front_wall", "back_wall"]
    index: int = Field(ge=0)
    time_us: float = Field(ge=0.0)
    depth_m: float | None = Field(default=None, ge=0.0)
    amplitude: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    time_std_us: float | None = Field(default=None, ge=0.0)

    @field_validator("amplitude", mode="before")
    @classmethod
    def parse_optional_echo_amplitude(cls, value: object) -> float | None:
        return _to_finite_float_or_none(value)


class NdtSignalAnalysisRecord(BaseModel):
    """Intermediate signal-analysis output reused across API responses."""

    total_peaks: int = Field(default=0, ge=0)
    peak_indices: list[int] = Field(default_factory=list)
    peak_times_us: list[float] = Field(default_factory=list)
    front_wall: NdtWallEchoRecord | None = None
    back_wall: NdtWallEchoRecord | None = None
    estimated_thickness_mm: float | None = Field(default=None, ge=0.0)
    thickness_std_mm: float | None = Field(default=None, ge=0.0)
    thickness_ci95_lower_mm: float | None = Field(default=None, ge=0.0)
    thickness_ci95_upper_mm: float | None = Field(default=None, ge=0.0)
    thickness_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    thickness_method: Literal["time_of_flight", "absolute_backwall", "insufficient_data"] = (
        "insufficient_data"
    )
    nominal_thickness_mm: float | None = Field(default=None, ge=0.0)
    thickness_error_mm: float | None = None
    thinning_flag: bool = False

    @model_validator(mode="after")
    def validate_peak_vectors(self) -> "NdtSignalAnalysisRecord":
        if len(self.peak_indices) != len(self.peak_times_us):
            raise ValueError("peak_indices and peak_times_us must have identical lengths")
        return self


class NdtSampleRecord(BaseModel):
    """In-memory representation of one NDT sample with validated numeric arrays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    path: Path
    rf: np.ndarray
    time: np.ndarray
    fs_hz: float = Field(gt=0.0)
    fc_hz: float = Field(gt=0.0)
    c_mps: float = Field(gt=0.0)
    thickness_m: float | None = None
    description: str
    defects: list[NdtDefectRecord] = Field(default_factory=list)

    @field_validator("rf", "time", mode="before")
    @classmethod
    def parse_arrays(cls, value: object) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("NDT arrays must contain at least one value")
        return arr

    @field_validator("thickness_m", mode="before")
    @classmethod
    def parse_thickness(cls, value: object) -> float | None:
        if value is None:
            return None
        parsed = _to_finite_float_or_none(value)
        if parsed is None:
            return None
        return parsed if parsed > 0 else None

    @model_validator(mode="after")
    def ensure_array_length_match(self) -> "NdtSampleRecord":
        if self.rf.size != self.time.size:
            raise ValueError("rf and time arrays must have identical lengths")
        return self


class BusiSampleRecord(BaseModel):
    """Validated BUSI sample payload returned from the repository layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    image_path: Path
    image_rgb: np.ndarray
    mask: np.ndarray

    @field_validator("image_rgb", mode="before")
    @classmethod
    def parse_rgb_image(cls, value: object) -> np.ndarray:
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BUSI image must be RGB with shape [H, W, 3]")
        return image

    @field_validator("mask", mode="before")
    @classmethod
    def parse_mask(cls, value: object) -> np.ndarray:
        mask = np.asarray(value, dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError("BUSI mask must be single-channel")
        return mask


class BusiTrainingSampleRecord(BaseModel):
    """One BUSI sample decoded from SQL storage for model training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: int = Field(ge=1)
    class_name: Literal["benign", "malignant", "normal"]
    label: int = Field(ge=0, le=2)
    split: Literal["train", "test"]
    image_rgb: np.ndarray

    @field_validator("image_rgb", mode="before")
    @classmethod
    def parse_training_rgb_image(cls, value: object) -> np.ndarray:
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BUSI training image must be RGB with shape [H, W, 3]")
        return image


class BusiTrainingCurvePointRecord(BaseModel):
    """One epoch point for train/test performance curves."""

    epoch: int = Field(ge=1)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)


class BusiTrainingRunRecord(BaseModel):
    """Persisted BUSI training run summary and learning curve."""

    run_id: int | None = Field(default=None, ge=1)
    created_at: datetime
    include_normal: bool = False
    epochs: int = Field(ge=1, le=200)
    batch_size: int = Field(ge=1, le=1024)
    learning_rate: float = Field(gt=0.0, le=1.0)
    train_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    class_counts: Dict[str, int]
    class_labels: list[str]
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)
    curve: list[BusiTrainingCurvePointRecord] = Field(default_factory=list)
    notes: str | None = None


class IndustrialTrainingSampleRecord(BaseModel):
    """One industrial sample decoded from SQL storage for model training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: int = Field(ge=1)
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    class_name: str
    label: int = Field(ge=0)
    split: Literal["train", "test"]
    image_rgb: np.ndarray
    annotation_blob: bytes | None = None

    @field_validator("image_rgb", mode="before")
    @classmethod
    def parse_industrial_training_rgb_image(cls, value: object) -> np.ndarray:
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Industrial training image must be RGB with shape [H, W, 3]")
        return image


class IndustrialTrainingCurvePointRecord(BaseModel):
    """One epoch point for industrial train/test curves."""

    epoch: int = Field(ge=1)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)


class IndustrialTrainingRunRecord(BaseModel):
    """Persisted industrial training run summary."""

    run_id: int | None = Field(default=None, ge=1)
    created_at: datetime
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    epochs: int = Field(ge=1, le=200)
    batch_size: int = Field(ge=1, le=1024)
    learning_rate: float = Field(gt=0.0, le=1.0)
    train_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    class_counts: Dict[str, int]
    class_labels: list[str]
    train_accuracy: float = Field(ge=0.0, le=1.0)
    test_accuracy: float = Field(ge=0.0, le=1.0)
    train_loss: float = Field(ge=0.0)
    test_loss: float = Field(ge=0.0)
    curve: list[IndustrialTrainingCurvePointRecord] = Field(default_factory=list)
    annotated_samples: int = Field(default=0, ge=0)
    segmentation_iou_train: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_iou_test: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_dice_train: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_dice_test: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class IndustrialSampleRecord(BaseModel):
    """Validated industrial defect sample from SQL storage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    split: str
    class_name: str
    requested_index: int = Field(ge=0)
    resolved_index: int = Field(ge=0)
    total_samples: int = Field(gt=0)
    relative_path: str
    image_rgb: np.ndarray
    annotation_blob: bytes | None = None
    has_annotation: bool = False

    @field_validator("image_rgb", mode="before")
    @classmethod
    def parse_industrial_rgb_image(cls, value: object) -> np.ndarray:
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Industrial sample image must be RGB with shape [H, W, 3]")
        return image


class BusiUploadRecord(BaseModel):
    """Metadata for one uploaded BUSI sample stored in SQL."""

    sample_id: int = Field(ge=1)
    class_name: Literal["benign", "malignant", "normal"]
    split: Literal["train", "test"]
    image_filename: str
    total_class_samples: int = Field(ge=1)
    created_at: datetime


class IndustrialUploadRecord(BaseModel):
    """Metadata for one uploaded industrial sample stored in SQL."""

    sample_id: int = Field(ge=1)
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"]
    split: str
    class_name: str
    image_filename: str
    relative_path: str
    has_annotation: bool = False
    total_class_samples: int = Field(ge=1)
    created_at: datetime


class JobRunRecord(BaseModel):
    """Background job state persisted in SQL."""

    id: int = Field(ge=1)
    job_type: Literal["busi_training", "dataset_resync", "industrial_training"]
    status: Literal["pending", "running", "completed", "failed"]
    requested_by: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AuthSessionRecord(BaseModel):
    """Authenticated API user context."""

    username: str
    role: Literal["viewer", "analyst", "admin"]
    expires_at: datetime


class ApiErrorEventRecord(BaseModel):
    """Captured API error event used by analytics service."""

    occurred_at: datetime
    request_id: str
    method: str
    path: str
    status_code: int = Field(ge=100, le=599)
    detail: str
    role: Literal["viewer", "analyst", "admin"] | None = None
