"""Liver Ultrasound Detection lab service.

Provides sample browsing, bbox annotation loading, and YOLO inference
for the Kaggle liver ultrasound dataset.  When fine-tuned weights exist
(from a training run), they are automatically used as the default model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    LiverDatasetStatusResponse,
    LiverSampleBbox,
    LiverSampleResponse,
    LiverYoloLabStatusResponse,
    YoloPredictRequest,
    YoloPredictResponse,
)
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.yolo_service import YoloService
from ultrasound.data.liver_dataset import (
    CLASS_NAMES,
    LiverDatasetPaths,
    load_annotations_csv,
    resolve_liver_paths,
    summarize_dataset,
)

logger = logging.getLogger("inphase.yolo.liver_lab")

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
        media_service: MediaService,
        yolo_service: YoloService,
    ) -> None:
        self._config = config
        self._media_service = media_service
        self._yolo_service = yolo_service

    def _paths(self) -> LiverDatasetPaths:
        return resolve_liver_paths(self._config.data_dir)

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
        return "yolo11n.pt"

    # -- Status ---------------------------------------------------------------

    def dataset_status(self) -> LiverDatasetStatusResponse:
        """Check whether the liver dataset is downloaded and parsed."""
        paths = self._paths()
        summary = summarize_dataset(paths) if paths.is_ready else {"status": "not_found"}
        return LiverDatasetStatusResponse(
            ready=paths.is_ready,
            summary=summary,
            generated_at=datetime.now(tz=timezone.utc),
        )

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

    # -- Sample browsing ------------------------------------------------------

    def _list_images_for_category(self, paths: LiverDatasetPaths, category: str) -> list[str]:
        """Return sorted image_ids (e.g., 'Benign_12') for a category."""
        flat_dir = paths.train_images_dir
        if not flat_dir.is_dir():
            return []
        prefix = category + "_"
        return sorted(
            f.stem for f in flat_dir.iterdir()
            if f.is_file() and f.stem.startswith(prefix)
        )

    def get_sample(self, category: str, sample_index: int) -> LiverSampleResponse:
        """Load a liver ultrasound sample by category and index."""
        paths = self._paths()
        if not paths.is_ready:
            raise FileNotFoundError(
                "Liver dataset not found. Download it first with "
                "scripts/download_liver_ultrasound_detection.py"
            )

        image_ids = self._list_images_for_category(paths, category)
        if not image_ids:
            raise FileNotFoundError(f"No images found for category '{category}'")

        resolved_index = sample_index % len(image_ids)
        image_id = image_ids[resolved_index]

        # Load image
        flat_dir = paths.train_images_dir
        candidates = list(flat_dir.glob(f"{image_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"Image file not found for {image_id}")
        img = Image.open(candidates[0]).convert("RGB")
        img_arr = np.asarray(img, dtype=np.uint8)

        # Load bboxes
        bboxes: list[LiverSampleBbox] = []
        csv_path = paths.annotations_csv
        if csv_path.is_file():
            annotations = load_annotations_csv(csv_path)
            for box in annotations.get(image_id, []):
                cls_id = int(box["class_id"])
                bboxes.append(LiverSampleBbox(
                    x_min=box["x_min"],
                    y_min=box["y_min"],
                    x_max=box["x_max"],
                    y_max=box["y_max"],
                    class_id=cls_id,
                    class_name=CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else None,
                ))

        return LiverSampleResponse(
            category=category,
            sample_index=resolved_index,
            total_samples=len(image_ids),
            image_id=image_id,
            image_shape=list(img_arr.shape),
            bboxes=bboxes,
            class_names=list(CLASS_NAMES),
            image_data_url=self._media_service.as_png_data_url(img_arr),
        )

    # -- Inference ------------------------------------------------------------

    def load_image_rgb(self, category: str, sample_index: int) -> np.ndarray:
        """Load a liver sample as an RGB numpy array for inference."""
        paths = self._paths()
        image_ids = self._list_images_for_category(paths, category)
        if not image_ids:
            raise FileNotFoundError(f"No images for category '{category}'")
        resolved_index = sample_index % len(image_ids)
        image_id = image_ids[resolved_index]
        candidates = list(paths.train_images_dir.glob(f"{image_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"Image file not found for {image_id}")
        return np.asarray(Image.open(candidates[0]).convert("RGB"), dtype=np.uint8)

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
        image_rgb = self.load_image_rgb(category, sample_index)

        # Auto-substitute trained weights when the user hasn't overridden.
        generic_models = {"yolo11n.pt", "yolov8n.pt", ""}
        if (request.model or "").strip() in generic_models:
            request = request.model_copy(update={"model": self.default_model})

        return self._yolo_service.predict(image_rgb=image_rgb, request=request)


