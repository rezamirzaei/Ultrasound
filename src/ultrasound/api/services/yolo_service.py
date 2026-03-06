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

    def _candidate_weights(self, requested_weights: str) -> tuple[str, ...]:
        requested = requested_weights.strip()
        if not requested:
            return self.DEFAULT_MODEL_CANDIDATES
        if requested in self.DEFAULT_MODEL_CANDIDATES:
            return (requested, *[item for item in self.DEFAULT_MODEL_CANDIDATES if item != requested])
        return (requested,)

    @staticmethod
    def _resolve_class_name(names: Any, class_id: int) -> str | None:
        if class_id < 0:
            return None
        if isinstance(names, dict):
            value = names.get(class_id)
            return str(value) if value is not None else None
        if isinstance(names, (list, tuple)) and class_id < len(names):
            return str(names[class_id])
        return None

    def predict(self, image_rgb: np.ndarray, request: YoloPredictRequest) -> YoloPredictResponse:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLO inference expects an RGB image with shape [H, W, 3]")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("YOLO inference expects a non-empty image")

        weights = (request.model or "").strip()
        model = None
        selected_weights = ""
        load_errors: list[str] = []
        fallback_notes: list[str] = []
        for candidate in self._candidate_weights(weights):
            try:
                model = self._load_model(candidate)
                selected_weights = candidate
                if weights and candidate != weights:
                    fallback_notes.append(
                        f"Requested model '{weights}' was unavailable; used fallback '{candidate}'."
                    )
                elif not weights and candidate != self.DEFAULT_MODEL_CANDIDATES[0]:
                    fallback_notes.append(f"Default model fallback selected '{candidate}'.")
                break
            except RuntimeError:
                raise
            except ValueError as exc:
                load_errors.append(str(exc))
        if model is None:
            raise ValueError(load_errors[0] if load_errors else "Failed to resolve a YOLO model.")

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
                model=selected_weights,
                image_shape=[int(image.shape[0]), int(image.shape[1]), 3],
                detections=[],
                annotated_image_data_url=self.media_service.as_png_data_url(image),
                backend="ultralytics",
                notes=" ".join([*fallback_notes, "No results returned by YOLO backend."]).strip(),
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
                class_name = self._resolve_class_name(names, class_id)

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
            model=selected_weights,
            image_shape=[int(image.shape[0]), int(image.shape[1]), 3],
            detections=detections,
            annotated_image_data_url=self.media_service.as_png_data_url(annotated_rgb),
            backend="ultralytics",
            notes=" ".join(fallback_notes) or None,
        )
