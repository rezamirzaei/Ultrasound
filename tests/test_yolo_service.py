"""Tests for the YOLO inference service."""

from __future__ import annotations

from typing import Any

import numpy as np

from ultrasound.api.models.schemas import YoloPredictRequest
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.yolo_service import YoloService


class _TensorArray:
    def __init__(self, values: list[list[float]] | list[float]) -> None:
        self._array = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _TensorArray:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _Boxes:
    def __init__(
        self,
        *,
        xyxy: list[list[float]],
        conf: list[float] | None = None,
        cls: list[float] | None = None,
    ) -> None:
        self.xyxy = _TensorArray(xyxy)
        self.conf = _TensorArray(conf) if conf is not None else None
        self.cls = _TensorArray(cls) if cls is not None else None


class _Result:
    def __init__(self, *, boxes: _Boxes | None, names: Any = None, plot_return: np.ndarray | None = None) -> None:
        self.boxes = boxes
        self.names = names
        self._plot_return = plot_return

    def plot(self) -> np.ndarray | None:
        return self._plot_return


class _Model:
    def __init__(self, result: _Result, *, names: Any = None) -> None:
        self._result = result
        self.names = names

    def predict(self, **_kwargs: Any) -> list[_Result]:
        return [self._result]


def test_predict_falls_back_between_default_models(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = _Result(boxes=None)
    fallback_model = _Model(result)

    def _load_model(weights: str) -> _Model:
        if weights == "yolo11n.pt":
            raise ValueError("missing yolo11n")
        if weights == "yolov8n.pt":
            return fallback_model
        raise AssertionError(f"unexpected weights {weights}")

    monkeypatch.setattr(service, "_load_model", _load_model)

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolo11n.pt"))

    assert response.model == "yolov8n.pt"
    assert response.notes is not None
    assert "fallback" in response.notes.lower()


def test_predict_resolves_class_names_from_sequences(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    plot_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = _Result(
        boxes=_Boxes(xyxy=[[2, 4, 10, 12]], conf=[0.9], cls=[1]),
        names=["liver", "mass"],
        plot_return=plot_bgr,
    )
    monkeypatch.setattr(service, "_load_model", lambda _weights: _Model(result))

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert len(response.detections) == 1
    assert response.detections[0].class_id == 1
    assert response.detections[0].class_name == "mass"


def test_predict_rejects_empty_images() -> None:
    service = YoloService(MediaService())

    try:
        service.predict(
            image_rgb=np.zeros((0, 0, 3), dtype=np.uint8),
            request=YoloPredictRequest(model="yolov8n.pt"),
        )
    except ValueError as exc:
        assert "non-empty image" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")
