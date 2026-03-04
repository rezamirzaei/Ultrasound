"""YOLO inference helpers using an optional Ultralytics backend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ultrasound.api.models.schemas import (
    YoloDetection,
    YoloPredictRequest,
    YoloPredictResponse,
    YoloStatusResponse,
    YoloXyxyBox,
)
from ultrasound.api.services.media_service import MediaService

logger = logging.getLogger("inphase.yolo")


class YoloService:
    """Run YOLO detections with a cached model instance."""

    DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = (
        # Prefer newer naming (if present in installed Ultralytics version),
        # then fall back to the widely-available YOLOv8 nano weights.
        "yolo11n.pt",
        "yolov8n.pt",
    )

    def __init__(self, media_service: MediaService):
        self.media_service = media_service
        self._models: dict[str, Any] = {}

    def backend_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401

            return True
        except Exception:
            return False

    def status(self) -> YoloStatusResponse:
        return YoloStatusResponse(
            available=self.backend_available(),
            backend="ultralytics",
            default_models=list(self.DEFAULT_MODEL_CANDIDATES),
        )

    def _load_model(self, weights: str) -> Any:
        if weights in self._models:
            return self._models[weights]

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - exercised via controller tests
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Install the optional dependency with "
                "`pip install -e '.[yolo]'` to enable YOLO inference."
            ) from exc

        try:
            model = YOLO(weights)
        except Exception as exc:
            raise ValueError(f"Failed to load YOLO model weights '{weights}': {exc}") from exc
        self._models[weights] = model
        return model

    def predict(self, image_rgb: np.ndarray, request: YoloPredictRequest) -> YoloPredictResponse:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLO inference expects an RGB image with shape [H, W, 3]")

        weights = (request.model or "").strip() or self.DEFAULT_MODEL_CANDIDATES[-1]
        model = self._load_model(weights)

        try:
            result_list = model.predict(
                source=image,
                conf=float(request.confidence),
                iou=float(request.iou_threshold),
                imgsz=int(request.image_size),
                max_det=int(request.max_detections),
                verbose=False,
            )
        except Exception as exc:
            raise ValueError(f"YOLO inference failed: {exc}") from exc
        if not result_list:
            return YoloPredictResponse(
                generated_at=datetime.now(tz=timezone.utc),
                model=weights,
                image_shape=[int(image.shape[0]), int(image.shape[1]), 3],
                detections=[],
                annotated_image_data_url=self.media_service.as_png_data_url(image),
                backend="ultralytics",
                notes="No results returned by YOLO backend.",
            )

        result = result_list[0]
        names = getattr(result, "names", None) or getattr(model, "names", None) or {}
        detections: list[YoloDetection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
            conf = (
                np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64)
                if boxes.conf is not None
                else None
            )
            cls = (
                np.asarray(boxes.cls.cpu().numpy(), dtype=np.float64)
                if boxes.cls is not None
                else None
            )
            n = int(xyxy.shape[0])

            for index in range(n):
                class_id = int(cls[index]) if cls is not None else -1
                class_name = None
                try:
                    if isinstance(names, dict) and class_id in names:
                        class_name = str(names[class_id])
                except Exception:
                    class_name = None

                bbox = YoloXyxyBox(
                    x1=float(xyxy[index, 0]),
                    y1=float(xyxy[index, 1]),
                    x2=float(xyxy[index, 2]),
                    y2=float(xyxy[index, 3]),
                )
                detections.append(
                    YoloDetection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf[index]) if conf is not None else 0.0,
                        bbox=bbox,
                    )
                )

        annotated_bgr = None
        try:
            annotated_bgr = result.plot()
        except Exception:
            annotated_bgr = None

        if annotated_bgr is None:
            annotated_rgb = image
        else:
            annotated_arr = np.asarray(annotated_bgr)
            if annotated_arr.ndim == 3 and annotated_arr.shape[2] == 3:
                annotated_rgb = annotated_arr[:, :, ::-1].astype(np.uint8, copy=False)
            else:
                annotated_rgb = image

        return YoloPredictResponse(
            generated_at=datetime.now(tz=timezone.utc),
            model=weights,
            image_shape=[int(image.shape[0]), int(image.shape[1]), 3],
            detections=detections,
            annotated_image_data_url=self.media_service.as_png_data_url(annotated_rgb),
            backend="ultralytics",
        )
