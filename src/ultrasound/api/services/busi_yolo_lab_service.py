"""BUSI-focused YOLO lab utilities (model download + inference + YOLO label derivation)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ultrasound.api.config import AppConfig
from ultrasound.api.models.domain import BusiSampleRecord
from ultrasound.api.models.schemas import (
    BusiSamplePreview,
    BusiYoloLabStatusResponse,
    BusiYoloModelStatus,
    BusiYoloSampleResponse,
    YoloLabel,
    YoloPredictRequest,
    YoloPredictResponse,
)
from ultrasound.api.services.interfaces import BusiSampleRepository, MediaRenderer, YoloPredictor
from ultrasound.api.services.service_errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    NotFoundError,
)
from ultrasound.api.services.yolo_lab_support import prefer_existing_model
from ultrasound.api.services.yolo_utils import (
    download_url_to_path,
    format_yolo_labels,
    load_manifest,
    mask_to_xyxy,
    write_manifest,
    xyxy_to_yolo_label,
)

logger = logging.getLogger("inphase.yolo.ultrasound")


class BusiYoloLabService:
    """YOLO lab helpers for BUSI (Breast Ultrasound Images) samples."""

    # Public BUSI YOLOv8 segmentation project with published weights.
    # Source project: https://github.com/sevdaimany/YOLOv8-Medical-Imaging
    RECOMMENDED_MODEL_ID = "busi_yolov8_seg"
    RECOMMENDED_MODEL_SOURCE_URL = (
        "https://raw.githubusercontent.com/sevdaimany/YOLOv8-Medical-Imaging/master/"
        "runs/segment/train/weights/best.pt"
    )
    RECOMMENDED_MODEL_FILENAME = "busi_yolov8_seg_best.pt"

    # Derived YOLO label class map for BUSI masks.
    YOLO_CLASS_NAMES: list[str] = ["benign", "malignant"]
    CLASS_TO_ID: dict[str, int] = {"benign": 0, "malignant": 1}

    def __init__(
        self,
        config: AppConfig,
        dataset_repository: BusiSampleRepository,
        media_service: MediaRenderer,
        yolo_service: YoloPredictor,
    ):
        self.config = config
        self.dataset_repository = dataset_repository
        self.media_service = media_service
        self.yolo_service = yolo_service

        self._root = self.config.artifacts_dir / "ultrasound_yolo" / "busi"
        self._models_dir = self._root / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def _model_path(self) -> Path:
        return self._models_dir / self.RECOMMENDED_MODEL_FILENAME

    def _manifest_path(self) -> Path:
        return self._models_dir / "model_manifest.json"

    @property
    def default_model(self) -> str:
        model_path = self._model_path()
        if model_path.exists():
            return str(model_path)
        default_candidates = tuple(self.yolo_service.DEFAULT_MODEL_CANDIDATES)
        if default_candidates:
            return default_candidates[0]
        return "yolo11n.pt"

    def model_status(self) -> BusiYoloModelStatus:
        model_path = self._model_path()
        downloaded = model_path.exists()
        manifest = load_manifest(self._manifest_path())

        size_bytes = None
        try:
            size_bytes = int(model_path.stat().st_size) if downloaded else None
        except OSError:
            size_bytes = None

        sha256 = manifest.sha256 if manifest is not None else None
        downloaded_at = manifest.downloaded_at if manifest is not None else None

        return BusiYoloModelStatus(
            model_id=self.RECOMMENDED_MODEL_ID,
            source_url=self.RECOMMENDED_MODEL_SOURCE_URL,
            local_path=str(model_path),
            downloaded=downloaded,
            sha256=sha256,
            size_bytes=size_bytes,
            downloaded_at=downloaded_at,
        )

    def status(self) -> BusiYoloLabStatusResponse:
        return BusiYoloLabStatusResponse(
            generated_at=datetime.now(tz=timezone.utc),
            yolo=self.yolo_service.status(),
            model=self.model_status(),
            yolo_class_names=list(self.YOLO_CLASS_NAMES),
        )

    def _load_sample(self, class_name: str, sample_index: int) -> BusiSampleRecord:
        try:
            return self.dataset_repository.get_busi_sample(class_name=class_name, index=sample_index)
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def download_recommended_model(self, force: bool = False) -> BusiYoloModelStatus:
        model_path = self._model_path()
        if model_path.exists() and not force:
            return self.model_status()

        url = self.RECOMMENDED_MODEL_SOURCE_URL
        logger.info("Downloading BUSI YOLO weights from %s -> %s", url, model_path)
        try:
            manifest = download_url_to_path(url, model_path)
        except Exception as exc:
            raise DependencyUnavailableError(
                f"Could not download recommended BUSI YOLO weights: {exc}"
            ) from exc
        try:
            write_manifest(self._manifest_path(), manifest)
        except OSError:
            logger.warning("Could not write BUSI YOLO model manifest to %s", self._manifest_path(), exc_info=True)

        return self.model_status()

    def get_sample(self, class_name: str, sample_index: int) -> BusiYoloSampleResponse:
        sample = self._load_sample(class_name=class_name, sample_index=sample_index)
        mask_binary = np.asarray(sample.mask > 0, dtype=np.uint8) * 255

        lesion_pixels = int(np.count_nonzero(mask_binary))
        lesion_ratio = float(lesion_pixels / mask_binary.size) if mask_binary.size else 0.0

        preview = BusiSamplePreview(
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

        bbox_xyxy = mask_to_xyxy(mask_binary)
        labels: list[YoloLabel] = []
        if bbox_xyxy is not None and sample.class_name in self.CLASS_TO_ID:
            class_id = int(self.CLASS_TO_ID[sample.class_name])
            label = xyxy_to_yolo_label(
                bbox=bbox_xyxy,
                class_id=class_id,
                class_name=sample.class_name,
                image_width=int(sample.image_rgb.shape[1]),
                image_height=int(sample.image_rgb.shape[0]),
            )
            labels.append(label)

        return BusiYoloSampleResponse(
            sample=preview,
            yolo_class_names=list(self.YOLO_CLASS_NAMES),
            yolo_labels=labels,
            raw_yolo_labels=format_yolo_labels(labels),
            bbox_xyxy=bbox_xyxy,
        )

    def predict(self, class_name: str, sample_index: int, request: YoloPredictRequest) -> YoloPredictResponse:
        recommended_path = self._model_path()
        request = prefer_existing_model(
            request,
            default_model_candidates=self.yolo_service.DEFAULT_MODEL_CANDIDATES,
            preferred_model_path=recommended_path,
            missing_explicit_preferred_message=(
                "Recommended BUSI YOLO weights are not downloaded yet. "
                "Call the model download endpoint first."
            ),
        )

        sample = self._load_sample(class_name=class_name, sample_index=sample_index)
        try:
            return self.yolo_service.predict(image_rgb=sample.image_rgb, request=request)
        except RuntimeError as exc:
            raise DependencyUnavailableError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
