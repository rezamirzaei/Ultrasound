"""Validated domain objects used across repositories and services."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
