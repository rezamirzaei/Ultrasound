"""Protocol definitions for injectable service collaborators."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np

from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
    BusiUploadRecord,
    IndustrialSampleRecord,
    IndustrialTrainingRunRecord,
    IndustrialTrainingSampleRecord,
    IndustrialUploadRecord,
    JobRunRecord,
    NdtAnalyzedDefect,
    NdtSampleRecord,
    NdtSignalAnalysisRecord,
)
from ultrasound.api.models.schemas import (
    BusiTrainingRequest,
    BusiTrainingResponse,
    DatasetResyncResponse,
    IndustrialTrainingRequest,
    IndustrialTrainingResponse,
    YoloPredictRequest,
    YoloPredictResponse,
    YoloStatusResponse,
)


class MediaRenderer(Protocol):
    def as_png_data_url(self, image: np.ndarray) -> str: ...


class YoloPredictor(Protocol):
    DEFAULT_MODEL_CANDIDATES: tuple[str, ...]

    def status(self) -> YoloStatusResponse: ...

    def predict(self, image_rgb: np.ndarray, request: YoloPredictRequest) -> YoloPredictResponse: ...


class BusiSampleRepository(Protocol):
    def get_busi_sample(self, class_name: str, index: int) -> BusiSampleRecord: ...


class BusiDatasetRepository(BusiSampleRepository, Protocol):
    def get_busi_counts(self) -> dict[str, int]: ...


class BusiTrainingRepository(Protocol):
    def list_busi_training_samples(self, include_normal: bool) -> list[BusiTrainingSampleRecord]: ...

    def get_latest_busi_training_run(self, include_normal: bool) -> BusiTrainingRunRecord | None: ...

    def get_busi_counts(self) -> dict[str, int]: ...

    def save_busi_training_run(self, run: BusiTrainingRunRecord) -> BusiTrainingRunRecord: ...


class BusiUploadRepository(Protocol):
    def add_busi_uploaded_sample(
        self,
        class_name: str,
        split: str,
        image_filename: str,
        image_blob: bytes,
        mask_blob: bytes | None,
    ) -> BusiUploadRecord: ...


class BusiSyncRepository(Protocol):
    def sync_busi_from_filesystem(self) -> int: ...


class NdtSampleRepository(Protocol):
    def list_ndt_samples(self) -> list[str]: ...

    def load_ndt_sample(self, sample_name: str) -> NdtSampleRecord: ...


class NdtSyncRepository(Protocol):
    def sync_ndt_from_filesystem(self) -> int: ...


class NdtDetectionAnalyzer(Protocol):
    def analyze_signal(self, sample: NdtSampleRecord) -> NdtSignalAnalysisRecord: ...

    def resolve_defects(
        self,
        sample: NdtSampleRecord,
        signal_analysis: NdtSignalAnalysisRecord | None = None,
    ) -> list[NdtAnalyzedDefect]: ...


class IndustrialSampleRepository(Protocol):
    def get_industrial_counts(self) -> dict[str, dict[str, dict[str, int]]]: ...

    def get_industrial_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        index: int,
    ) -> IndustrialSampleRecord: ...


class IndustrialTrainingRepository(IndustrialSampleRepository, Protocol):
    def list_industrial_training_samples(
        self, dataset_name: str
    ) -> tuple[list[IndustrialTrainingSampleRecord], dict[str, int], list[str]]: ...

    def get_latest_industrial_training_run(
        self, dataset_name: str
    ) -> IndustrialTrainingRunRecord | None: ...

    def get_industrial_counts(self) -> dict[str, dict[str, dict[str, int]]]: ...

    def get_industrial_annotation_count(self, dataset_name: str) -> int: ...

    def save_industrial_training_run(
        self, run: IndustrialTrainingRunRecord
    ) -> IndustrialTrainingRunRecord: ...

    def get_industrial_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        index: int,
    ) -> IndustrialSampleRecord: ...


class IndustrialUploadRepository(Protocol):
    def add_industrial_uploaded_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        image_filename: str,
        image_blob: bytes,
        annotation_blob: bytes | None,
    ) -> IndustrialUploadRecord: ...


class IndustrialSyncRepository(Protocol):
    def sync_industrial_from_filesystem(self) -> int: ...


class JobQueueRepository(Protocol):
    def enqueue(
        self,
        job_type: Literal["busi_training", "dataset_resync", "industrial_training"],
        requested_by: str,
        payload: dict[str, Any],
    ) -> JobRunRecord: ...

    def get_job(self, job_id: int) -> JobRunRecord | None: ...

    def list_jobs(self, limit: int = 50) -> list[JobRunRecord]: ...

    def claim_next_pending(self) -> JobRunRecord | None: ...

    def mark_completed(self, job_id: int, result: dict[str, Any]) -> JobRunRecord: ...

    def mark_failed(self, job_id: int, error_message: str) -> JobRunRecord: ...


class ObservabilityRecorder(Protocol):
    def set_worker_up(self, is_up: bool) -> None: ...

    def observe_job(self, job_type: str, status: str, duration_seconds: float) -> None: ...


class BusiTrainingRunner(Protocol):
    def run_training(self, request: BusiTrainingRequest) -> BusiTrainingResponse: ...


class IndustrialTrainingRunner(Protocol):
    def run_training(self, request: IndustrialTrainingRequest) -> IndustrialTrainingResponse: ...


class DatasetResyncRunner(Protocol):
    def resync_all(self) -> DatasetResyncResponse: ...
