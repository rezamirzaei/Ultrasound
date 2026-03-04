"""YOLO training pipeline for ultrasound detection tasks.

Provides a clean, reusable trainer that:
  1. Prepares data in Ultralytics YOLO format (images/ + labels/ per split).
  2. Fine-tunes a pretrained YOLO model on the prepared dataset.
  3. Runs validation and returns structured results.
"""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("inphase.yolo.trainer")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YoloTrainingConfig:
    """Hyperparameters and paths for a YOLO training run."""

    dataset_yaml: Path
    pretrained_weights: str = "yolo11n.pt"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    learning_rate: float = 0.01
    patience: int = 10
    device: str = ""  # "" = auto (MPS / CUDA / CPU)
    project_dir: Path = Path("outputs/yolo_runs")
    run_name: str = "liver_detection"
    exist_ok: bool = True
    workers: int = 2
    freeze_layers: int = 0  # number of backbone layers to freeze for fine-tuning

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass
class YoloTrainingResult:
    """Summary returned after a training run completes."""

    best_weights: Path | None = None
    last_weights: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    epochs_completed: int = 0
    run_dir: Path | None = None


# ---------------------------------------------------------------------------
# Dataset preparation (CSV bounding-box format → YOLO txt format)
# ---------------------------------------------------------------------------

class YoloDatasetPreparer:
    """Convert a detection dataset (images + CSV annotations) into YOLO format.

    Expected CSV columns: ``image_id, x_min, y_min, x_max, y_max, class_id``.
    Supports any number of classes.
    """

    def __init__(
        self,
        source_images_dir: Path,
        annotations_csv: Path,
        output_dir: Path,
        class_names: list[str],
        *,
        train_ratio: float = 0.8,
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        seed: int = 42,
    ) -> None:
        self.source_images_dir = Path(source_images_dir)
        self.annotations_csv = Path(annotations_csv)
        self.output_dir = Path(output_dir)
        self.class_names = list(class_names)
        self.train_ratio = train_ratio
        self.image_extensions = image_extensions
        self.seed = seed

    # -- public API ----------------------------------------------------------

    def prepare(self) -> Path:
        """Build the YOLO dataset directory and return the path to ``data.yaml``."""
        logger.info("Preparing YOLO dataset in %s", self.output_dir)

        annotations = self._load_annotations()
        image_ids = sorted(annotations.keys())
        if not image_ids:
            raise FileNotFoundError(
                f"No annotated images found. CSV={self.annotations_csv}, "
                f"images_dir={self.source_images_dir}"
            )

        train_ids, val_ids = self._split(image_ids)
        logger.info("Split: %d train / %d val", len(train_ids), len(val_ids))

        for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
            self._write_split(split_name, split_ids, annotations)

        yaml_path = self._write_data_yaml()
        logger.info("Dataset YAML: %s", yaml_path)
        return yaml_path

    # -- internals -----------------------------------------------------------

    def _load_annotations(self) -> dict[str, list[dict[str, Any]]]:
        """Parse CSV and group rows by image_id."""
        annotations: dict[str, list[dict[str, Any]]] = {}
        with self.annotations_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_id = row["image_id"].strip()
                annotations.setdefault(image_id, []).append({
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                    "class_id": int(row.get("class_id", 0)),
                })
        return annotations

    def _split(self, image_ids: list[str]) -> tuple[list[str], list[str]]:
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(image_ids))
        n_train = max(1, int(len(image_ids) * self.train_ratio))
        train = [image_ids[i] for i in indices[:n_train]]
        val = [image_ids[i] for i in indices[n_train:]]
        return train, val

    def _find_image(self, image_id: str) -> Path | None:
        for ext in self.image_extensions:
            candidate = self.source_images_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        # Try with the image_id as-is (maybe it already has extension).
        candidate = self.source_images_dir / image_id
        if candidate.exists():
            return candidate
        return None

    def _write_split(
        self,
        split_name: str,
        image_ids: list[str],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        images_dir = self.output_dir / split_name / "images"
        labels_dir = self.output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            src_path = self._find_image(image_id)
            if src_path is None:
                logger.warning("Image not found for id=%s, skipping.", image_id)
                continue

            dest_image = images_dir / src_path.name
            shutil.copy2(src_path, dest_image)

            with Image.open(dest_image) as img:
                img_w, img_h = img.size

            label_lines: list[str] = []
            for ann in annotations.get(image_id, []):
                label_lines.append(
                    self._xyxy_to_yolo_line(ann, img_w, img_h)
                )

            label_path = labels_dir / f"{src_path.stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _xyxy_to_yolo_line(ann: dict[str, Any], img_w: int, img_h: int) -> str:
        """Convert xyxy pixel coords to YOLO normalized xywh format."""
        x_min, y_min = float(ann["x_min"]), float(ann["y_min"])
        x_max, y_max = float(ann["x_max"]), float(ann["y_max"])
        class_id = int(ann["class_id"])

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.001, min(1.0, width))
        height = max(0.001, min(1.0, height))

        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _write_data_yaml(self) -> Path:
        yaml_path = self.output_dir / "data.yaml"
        lines = [
            f"path: {self.output_dir.resolve()}",
            "train: train/images",
            "val: val/images",
            "",
            f"nc: {len(self.class_names)}",
            f"names: {self.class_names}",
        ]
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class YoloTrainer:
    """Fine-tune or train a YOLO detection model using Ultralytics."""

    def __init__(self) -> None:
        self._ensure_backend()

    @staticmethod
    def _ensure_backend() -> None:
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

    def train(self, config: YoloTrainingConfig) -> YoloTrainingResult:
        """Execute a YOLO training run and return structured results."""
        from ultralytics import YOLO

        logger.info(
            "Starting YOLO training: weights=%s epochs=%d imgsz=%d batch=%d",
            config.pretrained_weights,
            config.epochs,
            config.image_size,
            config.batch_size,
        )

        model = YOLO(config.pretrained_weights)

        # Freeze backbone layers for fine-tuning if requested.
        if config.freeze_layers > 0:
            self._freeze_backbone(model, config.freeze_layers)

        train_kwargs: dict[str, Any] = {
            "data": str(config.dataset_yaml),
            "epochs": config.epochs,
            "imgsz": config.image_size,
            "batch": config.batch_size,
            "lr0": config.learning_rate,
            "patience": config.patience,
            "project": str(config.project_dir),
            "name": config.run_name,
            "exist_ok": config.exist_ok,
            "workers": config.workers,
            "verbose": True,
        }
        if config.device:
            train_kwargs["device"] = config.device

        results = model.train(**train_kwargs)

        # Resolve the actual run directory.  Ultralytics may relocate it
        # (e.g. under ``runs/detect/``) when the project path is relative.
        run_dir = Path(config.project_dir) / config.run_name
        save_dir = getattr(results, "save_dir", None)
        if save_dir is not None:
            run_dir = Path(str(save_dir))

        best_weights = run_dir / "weights" / "best.pt"
        last_weights = run_dir / "weights" / "last.pt"

        metrics = self._extract_metrics(results)

        return YoloTrainingResult(
            best_weights=best_weights if best_weights.exists() else None,
            last_weights=last_weights if last_weights.exists() else None,
            metrics=metrics,
            epochs_completed=config.epochs,
            run_dir=run_dir,
        )

    def validate(self, weights_path: Path, data_yaml: Path, image_size: int = 640) -> dict[str, float]:
        """Run validation on a trained model and return metrics."""
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        results = model.val(data=str(data_yaml), imgsz=image_size, verbose=False)
        return self._extract_metrics(results)

    @staticmethod
    def _freeze_backbone(model: Any, n_layers: int) -> None:
        """Freeze the first *n_layers* of the model backbone."""
        try:
            for i, (name, param) in enumerate(model.model.named_parameters()):
                if i < n_layers:
                    param.requires_grad = False
                    logger.debug("Frozen: %s", name)
        except Exception:
            logger.warning("Could not freeze backbone layers – skipping.")

    @staticmethod
    def _extract_metrics(results: Any) -> dict[str, float]:
        """Pull numeric metrics from an Ultralytics results object."""
        metrics: dict[str, float] = {}
        if results is None:
            return metrics

        # Ultralytics stores metrics in results.results_dict or results.box
        results_dict = getattr(results, "results_dict", None)
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    pass
            return metrics

        # Fallback: try box attribute
        box = getattr(results, "box", None)
        if box is not None:
            for attr in ("map50", "map", "mp", "mr"):
                val = getattr(box, attr, None)
                if val is not None:
                    try:
                        metrics[attr] = float(val)
                    except (TypeError, ValueError):
                        pass

        return metrics

