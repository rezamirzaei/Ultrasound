"""Tests for the YOLO inference service."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ultrasound.api.models.schemas import YoloPredictRequest
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.yolo_service import YoloService


class _TensorArray:
    def __init__(self, values: Sequence[Sequence[float]] | Sequence[float]) -> None:
        self._array = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _TensorArray:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _Boxes:
    def __init__(
        self,
        *,
        xyxy: Sequence[Sequence[float]] | Sequence[float],
        conf: Sequence[float] | None = None,
        cls: Sequence[float] | None = None,
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
        self.last_predict_kwargs: dict[str, Any] | None = None

    def predict(self, **_kwargs: Any) -> list[_Result]:
        self.last_predict_kwargs = _kwargs
        return [self._result]


class _EmptyModel:
    def predict(self, **_kwargs: Any) -> list[_Result]:
        return []


class _FailingModel:
    def predict(self, **_kwargs: Any) -> list[_Result]:
        raise RuntimeError("backend exploded")


class _PlotErrorResult(_Result):
    def plot(self) -> np.ndarray | None:
        raise RuntimeError("plot failed")


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


def test_backend_available_returns_false_when_ultralytics_import_fails(monkeypatch) -> None:
    service = YoloService(MediaService())
    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ultralytics":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    assert service.backend_available() is False


def test_load_model_caches_instances(monkeypatch) -> None:
    service = YoloService(MediaService())
    calls: list[str] = []

    def _ctor(weights: str) -> object:
        calls.append(weights)
        return object()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_ctor))

    first = service._load_model("weights.pt")
    second = service._load_model("weights.pt")

    assert first is second
    assert calls == ["weights.pt"]


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


def test_predict_rejects_invalid_image_shape() -> None:
    service = YoloService(MediaService())

    with pytest.raises(ValueError, match="RGB image"):
        service.predict(
            image_rgb=np.zeros((8, 8), dtype=np.uint8),
            request=YoloPredictRequest(model="yolov8n.pt"),
        )


def test_predict_normalizes_float_images_before_backend(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.full((8, 8, 3), 300.0, dtype=np.float32)
    image[0, 0, 0] = -10.0
    model = _Model(_Result(boxes=None))
    monkeypatch.setattr(service, "_load_model", lambda _weights: model)

    service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert model.last_predict_kwargs is not None
    source = model.last_predict_kwargs["source"]
    assert source.dtype == np.uint8
    assert int(source.min()) == 0
    assert int(source.max()) == 255


def test_predict_scales_unit_range_float_images(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    model = _Model(_Result(boxes=None))
    monkeypatch.setattr(service, "_load_model", lambda _weights: model)

    service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert model.last_predict_kwargs is not None
    source = model.last_predict_kwargs["source"]
    assert 126 <= int(source[0, 0, 0]) <= 128


def test_predict_rejects_nonfinite_images() -> None:
    service = YoloService(MediaService())
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan

    with np.testing.assert_raises_regex(ValueError, "finite image values"):
        service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))


def test_predict_uses_default_candidates_when_request_model_is_blank(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    calls: list[str] = []

    def _load_model(weights: str) -> _Model:
        calls.append(weights)
        if weights == service.DEFAULT_MODEL_CANDIDATES[0]:
            raise ValueError("missing primary default")
        return _Model(_Result(boxes=None))

    monkeypatch.setattr(service, "_load_model", _load_model)

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="   "))

    assert calls[:2] == list(service.DEFAULT_MODEL_CANDIDATES)
    assert response.model == service.DEFAULT_MODEL_CANDIDATES[1]
    assert response.notes is not None
    assert "default model fallback" in response.notes.lower()


def test_predict_clips_bboxes_and_skips_invalid_rows(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    result = _Result(
        boxes=_Boxes(
            xyxy=[
                [-5.0, -1.0, 50.0, 60.0],
                [1.0, 2.0, float("nan"), 4.0],
                [8.0, 9.0, 4.0, 5.0],
            ],
            conf=[0.7, 0.5, np.nan],
            cls=[0, 1, 2],
        ),
        names={0: "lesion"},
    )
    monkeypatch.setattr(service, "_load_model", lambda _weights: _Model(result))

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert len(response.detections) == 1
    detection = response.detections[0]
    assert detection.class_name == "lesion"
    assert detection.bbox.x1 == 0.0
    assert detection.bbox.y1 == 0.0
    assert detection.bbox.x2 == 11.0
    assert detection.bbox.y2 == 9.0


def test_predict_returns_empty_response_when_backend_returns_no_results(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(service, "_load_model", lambda _weights: _EmptyModel())

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert response.detections == []
    assert response.annotated_image_data_url == service.media_service.as_png_data_url(image)
    assert response.notes is not None
    assert "No results returned" in response.notes


def test_predict_raises_when_backend_returns_invalid_detection_coordinates(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = _Result(
        boxes=_Boxes(xyxy=[1, 2, 3, 4], conf=[0.8], cls=[0]),
        names={0: "lesion"},
    )
    monkeypatch.setattr(service, "_load_model", lambda _weights: _Model(result))

    with pytest.raises(ValueError, match="invalid detection coordinates"):
        service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))


def test_predict_wraps_backend_inference_errors(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(service, "_load_model", lambda _weights: _FailingModel())

    with pytest.raises(ValueError, match="YOLO inference failed: backend exploded"):
        service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))


def test_predict_falls_back_to_original_image_when_plot_fails(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = _PlotErrorResult(
        boxes=_Boxes(xyxy=[[1, 1, 4, 4]], conf=[0.6], cls=[-1]),
        names={0: "lesion"},
    )
    monkeypatch.setattr(service, "_load_model", lambda _weights: _Model(result))

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert response.detections[0].class_name is None
    assert response.annotated_image_data_url == service.media_service.as_png_data_url(image)


def test_predict_falls_back_to_original_image_for_non_rgb_plot(monkeypatch) -> None:
    service = YoloService(MediaService())
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = _Result(
        boxes=None,
        plot_return=np.zeros((10, 10), dtype=np.uint8),
    )
    monkeypatch.setattr(service, "_load_model", lambda _weights: _Model(result))

    response = service.predict(image_rgb=image, request=YoloPredictRequest(model="yolov8n.pt"))

    assert response.annotated_image_data_url == service.media_service.as_png_data_url(image)
