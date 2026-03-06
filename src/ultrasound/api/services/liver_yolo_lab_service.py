"""Liver Ultrasound Detection lab service.

Provides sample browsing, bbox annotation loading, and YOLO inference
for the Kaggle liver ultrasound dataset.  When fine-tuned weights exist
(from a training run), they are automatically used as the default model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    LiverDatasetStatusResponse,
    LiverSampleResponse,
    LiverYoloLabStatusResponse,
    YoloPredictRequest,
    YoloPredictResponse,
)
from ultrasound.api.services.interfaces import MediaRenderer, YoloPredictor
from ultrasound.api.services.liver_dataset_browser import LiverDatasetBrowser
from ultrasound.api.services.service_errors import DependencyUnavailableError, InvalidRequestError
from ultrasound.api.services.yolo_lab_support import prefer_existing_model
from ultrasound.data.liver_dataset import CLASS_NAMES

# Well-known locations where training saves best weights (checked in order).
_TRAINED_WEIGHTS_CANDIDATES: tuple[str, ...] = (
    "models/liver_yolo_best.pt",
    "outputs/yolo_runs/liver_detection/weights/best.pt",
    "outputs/api/yolo_runs/liver_detection/weights/best.pt",
    "runs/detect/outputs/yolo_runs/liver_detection/weights/best.pt",
)


class LiverYoloLabService:
    """Browse liver ultrasound samples and run YOLO inference."""

    def __init__(
        self,
        config: AppConfig,
        media_service: MediaRenderer,
        yolo_service: YoloPredictor,
        dataset_browser: LiverDatasetBrowser | None = None,
    ) -> None:
        self._config = config
        self._media_service = media_service
        self._yolo_service = yolo_service
        self._dataset_browser = dataset_browser or LiverDatasetBrowser(config)

    # -- Trained weights resolution -------------------------------------------

    def resolve_trained_weights(self) -> Path | None:
        """Return the path to fine-tuned liver YOLO weights, or *None*."""
        root = self._config.project_root
        for candidate in _TRAINED_WEIGHTS_CANDIDATES:
            path = root / candidate
            if path.is_file():
                return path
        return None

    @property
    def default_model(self) -> str:
        """Best available model: fine-tuned weights when present, else generic."""
        weights = self.resolve_trained_weights()
        if weights is not None:
            return str(weights)
        return self._yolo_service.DEFAULT_MODEL_CANDIDATES[0]

    # -- Status ---------------------------------------------------------------

    def dataset_status(self) -> LiverDatasetStatusResponse:
        """Check whether the liver dataset is downloaded and parsed."""
        return self._dataset_browser.dataset_status()

    def lab_status(self) -> LiverYoloLabStatusResponse:
        """Combined YOLO backend + dataset readiness."""
        trained = self.resolve_trained_weights()
        return LiverYoloLabStatusResponse(
            generated_at=datetime.now(tz=timezone.utc),
            yolo=self._yolo_service.status(),
            dataset=self.dataset_status(),
            class_names=list(CLASS_NAMES),
            trained_weights=str(trained) if trained else None,
            default_model=self.default_model,
        )

    def get_sample(self, category: str, sample_index: int) -> LiverSampleResponse:
        """Load a liver ultrasound sample by category and index."""
        sample = self._dataset_browser.load_sample(category, sample_index)

        return LiverSampleResponse(
            category=sample.category,
            sample_index=sample.sample_index,
            total_samples=sample.total_samples,
            image_id=sample.image_id,
            image_shape=list(sample.image_rgb.shape),
            bboxes=sample.bboxes,
            class_names=list(CLASS_NAMES),
            image_data_url=self._media_service.as_png_data_url(sample.image_rgb),
        )

    def predict(
        self,
        category: str,
        sample_index: int,
        request: YoloPredictRequest,
    ) -> YoloPredictResponse:
        """Run YOLO inference on a liver sample.

        When the request does not specify a model (or uses the generic
        pretrained weights), automatically substitute fine-tuned liver
        weights if they are available.
        """
        sample = self._dataset_browser.load_sample(category, sample_index)

        request = prefer_existing_model(
            request,
            default_model_candidates=self._yolo_service.DEFAULT_MODEL_CANDIDATES,
            preferred_model_path=self.resolve_trained_weights(),
        )
        try:
            return self._yolo_service.predict(image_rgb=sample.image_rgb, request=request)
        except RuntimeError as exc:
            raise DependencyUnavailableError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
