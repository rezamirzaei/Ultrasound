"""Liver dataset browsing and sample loading for the YOLO lab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    LiverDatasetStatusResponse,
    LiverSampleBbox,
)
from ultrasound.api.services.service_errors import NotFoundError
from ultrasound.data.liver_dataset import (
    CLASS_NAMES,
    LiverDatasetPaths,
    load_annotations_csv,
    resolve_liver_paths,
    summarize_dataset,
)


@dataclass(frozen=True)
class LiverLoadedSample:
    category: str
    sample_index: int
    total_samples: int
    image_id: str
    image_rgb: np.ndarray
    bboxes: list[LiverSampleBbox]


class LiverDatasetBrowser:
    """Read liver YOLO lab samples from disk with consistent validation."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def dataset_status(self) -> LiverDatasetStatusResponse:
        paths = self._paths()
        summary = summarize_dataset(paths) if paths.is_ready else {"status": "not_found"}
        return LiverDatasetStatusResponse(
            ready=paths.is_ready,
            summary=summary,
            generated_at=datetime.now(tz=timezone.utc),
        )

    def load_sample(self, category: str, sample_index: int) -> LiverLoadedSample:
        paths = self._require_ready_paths()
        resolved_category = self._resolve_category(category)
        image_ids = self._list_images_for_category(paths, resolved_category)
        if not image_ids:
            raise NotFoundError(f"No images found for category '{resolved_category}'")

        resolved_index = sample_index % len(image_ids)
        image_id = image_ids[resolved_index]
        image_rgb = self._load_image_rgb(paths, image_id)
        bboxes = self._load_bboxes(paths, image_id)
        return LiverLoadedSample(
            category=resolved_category,
            sample_index=resolved_index,
            total_samples=len(image_ids),
            image_id=image_id,
            image_rgb=image_rgb,
            bboxes=bboxes,
        )

    def _paths(self) -> LiverDatasetPaths:
        return resolve_liver_paths(self._config.data_dir)

    def _require_ready_paths(self) -> LiverDatasetPaths:
        paths = self._paths()
        if not paths.is_ready:
            raise NotFoundError(
                "Liver dataset not found. Download it first with "
                "scripts/download_liver_ultrasound_detection.py"
            )
        return paths

    @staticmethod
    def _resolve_category(category: str) -> str:
        normalized = category.strip().lower()
        for candidate in ("Benign", "Malignant", "Normal"):
            if candidate.lower() == normalized:
                return candidate
        raise NotFoundError(f"Unknown liver category '{category}'")

    def _list_images_for_category(self, paths: LiverDatasetPaths, category: str) -> list[str]:
        flat_dir = paths.train_images_dir
        if not flat_dir.is_dir():
            return []

        prefix = category + "_"
        image_ids = sorted(
            image_path.stem
            for image_path in flat_dir.iterdir()
            if image_path.is_file() and image_path.stem.startswith(prefix)
        )
        if image_ids:
            return image_ids

        has_prefixed_ids = any(
            image_path.is_file()
            and any(
                image_path.stem.startswith(f"{candidate}_")
                for candidate in ("Benign", "Malignant", "Normal")
            )
            for image_path in flat_dir.iterdir()
        )
        if has_prefixed_ids:
            return []
        return sorted(image_path.stem for image_path in flat_dir.iterdir() if image_path.is_file())

    @staticmethod
    def _load_image_rgb(paths: LiverDatasetPaths, image_id: str) -> np.ndarray:
        candidates = list(paths.train_images_dir.glob(f"{image_id}.*"))
        if not candidates:
            raise NotFoundError(f"Image file not found for {image_id}")
        return np.asarray(Image.open(candidates[0]).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _load_bboxes(paths: LiverDatasetPaths, image_id: str) -> list[LiverSampleBbox]:
        if not paths.annotations_csv.is_file():
            return []
        annotations = load_annotations_csv(paths.annotations_csv)
        bboxes: list[LiverSampleBbox] = []
        for box in annotations.get(image_id, []):
            class_id = int(box["class_id"])
            bboxes.append(
                LiverSampleBbox(
                    x_min=box["x_min"],
                    y_min=box["y_min"],
                    x_max=box["x_max"],
                    y_max=box["y_max"],
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else None,
                )
            )
        return bboxes
