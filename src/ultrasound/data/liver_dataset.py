"""Liver Ultrasound Detection dataset utilities.

Handles downloading, parsing, and preparing the Kaggle
*Annotated Ultrasound Liver Images Dataset* for YOLO detection training.

Dataset: https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset

The dataset provides:
  - ``{Benign,Malignant,Normal}/image/``          — ultrasound JPGs
  - ``{Benign,Malignant,Normal}/segmentation/liver/`` — liver polygon JSONs
  - ``{Benign,Malignant,Normal}/segmentation/mass/``  — mass polygon JSONs (Benign/Malignant only)

For **detection** we derive tight bounding boxes from the polygon annotations.
Classes: 0=liver, 1=mass
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("inphase.yolo.liver")

# Detection classes derived from the polygon annotations.
CLASS_NAMES: list[str] = ["liver", "mass"]

# Public Kaggle dataset URL (no auth needed).
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "orvile/annotated-ultrasound-liver-images-dataset"
)

# The ZIP extracts everything under a numbered subfolder.
_INNER_DIR = "7272660"
_CATEGORIES = ("Benign", "Malignant", "Normal")


@dataclass(frozen=True)
class LiverDatasetPaths:
    """Resolved paths for the liver ultrasound detection dataset."""

    root: Path

    @property
    def data_root(self) -> Path:
        """Path to the inner numbered directory containing {Benign,Malignant,Normal}."""
        inner = self.root / _INNER_DIR
        if inner.is_dir():
            return inner
        return self.root

    @property
    def is_ready(self) -> bool:
        dr = self.data_root
        if any((dr / cat / cat / "image").is_dir() for cat in _CATEGORIES):
            return True
        return self.train_images_dir.is_dir() and self.annotations_csv.is_file()

    @property
    def annotations_csv(self) -> Path:
        return self.root / "annotations.csv"

    @property
    def train_images_dir(self) -> Path:
        """Convenience — points to the flat images dir created by prepare_flat_images."""
        return self.root / "images_flat"


def resolve_liver_paths(data_dir: Path) -> LiverDatasetPaths:
    """Build expected paths under ``data_dir/liver_ultrasound_detection/``."""
    return LiverDatasetPaths(root=data_dir / "liver_ultrasound_detection")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_liver_dataset(dest_dir: Path, *, force: bool = False) -> LiverDatasetPaths:
    """Download the liver ultrasound dataset from Kaggle (no auth required).

    The dataset (~74 MB) is downloaded, extracted, and polygon annotations
    are converted to a bounding-box CSV.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = LiverDatasetPaths(root=dest_dir)

    if paths.is_ready and not force:
        logger.info("Dataset already present at %s", dest_dir)
        _ensure_annotations_csv(paths)
        _ensure_flat_images(paths)
        return paths

    zip_path = dest_dir / "dataset.zip"

    # Download
    if not zip_path.exists() or force:
        logger.info("Downloading liver ultrasound dataset (~74 MB) ...")
        _download_file(DATASET_URL, zip_path)
        logger.info("Downloaded %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)

    # Extract
    logger.info("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted to %s", dest_dir)

    # Build CSV annotations from polygon JSONs
    _ensure_annotations_csv(paths)
    _ensure_flat_images(paths)
    return paths


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "inPhase-ultrasound-toolkit/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.info("  %.0f%% (%.1f / %.1f MB)", pct, downloaded / 1e6, total / 1e6)


# ---------------------------------------------------------------------------
# Convert polygon JSONs → bounding-box CSV
# ---------------------------------------------------------------------------

def _extract_polygon_points(payload: object) -> list[tuple[float, float]]:
    """Normalize supported JSON payload shapes into 2-D polygon points."""
    if isinstance(payload, dict):
        for key in ("points", "polygon", "segmentation"):
            if key in payload:
                return _extract_polygon_points(payload[key])
        shapes = payload.get("shapes")
        if isinstance(shapes, list):
            for shape in shapes:
                try:
                    return _extract_polygon_points(shape)
                except ValueError:
                    continue
        raise ValueError("polygon payload does not contain point coordinates")

    if not isinstance(payload, list) or not payload:
        raise ValueError("polygon payload must be a non-empty list")

    if len(payload) == 1 and isinstance(payload[0], (list, dict)):
        return _extract_polygon_points(payload[0])

    points: list[tuple[float, float]] = []
    for point in payload:
        if not isinstance(point, list) or len(point) < 2:
            raise ValueError("polygon points must be [x, y] pairs")
        x = float(point[0])
        y = float(point[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        points.append((x, y))

    if len(points) < 2:
        raise ValueError("polygon must contain at least two points")
    return points


def _polygon_to_bbox(polygon: object) -> tuple[float, float, float, float]:
    """Convert a polygon payload into (x_min, y_min, x_max, y_max)."""
    points = _extract_polygon_points(polygon)
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        raise ValueError("polygon bounding box must have positive area")
    return min(xs), min(ys), max(xs), max(ys)


def _ensure_annotations_csv(paths: LiverDatasetPaths) -> Path:
    """Build ``annotations.csv`` from the polygon JSON files."""
    csv_path = paths.annotations_csv
    if csv_path.exists():
        return csv_path

    logger.info("Building annotations.csv from polygon JSONs ...")
    data_root = paths.data_root
    rows: list[dict[str, str]] = []

    for category in _CATEGORIES:
        cat_dir = data_root / category / category
        image_dir = cat_dir / "image"
        liver_dir = cat_dir / "segmentation" / "liver"
        mass_dir = cat_dir / "segmentation" / "mass"

        if not image_dir.is_dir():
            continue

        for img_path in sorted(image_dir.glob("*")):
            if not img_path.is_file():
                continue
            stem = img_path.stem
            image_id = f"{category}_{stem}"

            # Liver bbox (class_id=0)
            liver_json = liver_dir / f"{stem}.json"
            if liver_json.exists():
                try:
                    polygon = json.loads(liver_json.read_text(encoding="utf-8"))
                    x_min, y_min, x_max, y_max = _polygon_to_bbox(polygon)
                    rows.append({
                        "image_id": image_id,
                        "x_min": f"{x_min:.2f}",
                        "y_min": f"{y_min:.2f}",
                        "x_max": f"{x_max:.2f}",
                        "y_max": f"{y_max:.2f}",
                        "class_id": "0",
                        "category": category,
                    })
                except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
                    logger.warning("Skipping bad liver JSON: %s", liver_json)

            # Mass bbox (class_id=1) — only Benign/Malignant have masses
            mass_json = mass_dir / f"{stem}.json"
            if mass_json.exists():
                try:
                    polygon = json.loads(mass_json.read_text(encoding="utf-8"))
                    x_min, y_min, x_max, y_max = _polygon_to_bbox(polygon)
                    rows.append({
                        "image_id": image_id,
                        "x_min": f"{x_min:.2f}",
                        "y_min": f"{y_min:.2f}",
                        "x_max": f"{x_max:.2f}",
                        "y_max": f"{y_max:.2f}",
                        "class_id": "1",
                        "category": category,
                    })
                except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
                    logger.warning("Skipping bad mass JSON: %s", mass_json)

    fieldnames = ["image_id", "x_min", "y_min", "x_max", "y_max", "class_id", "category"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d annotation rows to %s", len(rows), csv_path)
    return csv_path


def _ensure_flat_images(paths: LiverDatasetPaths) -> Path:
    """Copy images into a single flat directory with unique names.

    The YOLO preparer expects all images in one folder.
    Creates ``images_flat/`` with files like ``Benign_63.jpg``.
    """

    flat_dir = paths.train_images_dir
    if flat_dir.is_dir() and any(flat_dir.iterdir()):
        return flat_dir

    flat_dir.mkdir(parents=True, exist_ok=True)
    data_root = paths.data_root
    count = 0

    for category in _CATEGORIES:
        image_dir = data_root / category / category / "image"
        if not image_dir.is_dir():
            continue
        for img_path in sorted(image_dir.glob("*")):
            if not img_path.is_file():
                continue
            dest = flat_dir / f"{category}_{img_path.name}"
            shutil.copy2(img_path, dest)
            count += 1

    logger.info("Copied %d images to %s", count, flat_dir)
    return flat_dir


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotations_csv(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    """Parse ``annotations.csv`` → ``{image_id: [{x_min, y_min, x_max, y_max, class_id}]}``."""
    annotations: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                image_id = row["image_id"].strip()
                x_min = float(row["x_min"])
                y_min = float(row["y_min"])
                x_max = float(row["x_max"])
                y_max = float(row["y_max"])
                class_id = int(row.get("class_id", 0))
            except (AttributeError, KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed annotation row in %s", csv_path)
                continue

            if not image_id:
                logger.warning("Skipping annotation row with empty image_id in %s", csv_path)
                continue
            if not all(np.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
                logger.warning("Skipping non-finite annotation row for %s", image_id)
                continue
            if x_max <= x_min or y_max <= y_min or class_id < 0:
                logger.warning("Skipping invalid annotation bounds for %s", image_id)
                continue

            annotations.setdefault(image_id, []).append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "class_id": class_id,
            })
    return annotations


def summarize_dataset(paths: LiverDatasetPaths) -> dict[str, int | str]:
    """Quick summary of the dataset on disk."""
    summary: dict[str, int | str] = {"root": str(paths.root)}

    flat_dir = paths.train_images_dir
    if flat_dir.is_dir():
        summary["images"] = sum(1 for f in flat_dir.iterdir() if f.is_file())
    else:
        # Count from nested structure
        total = 0
        for cat in _CATEGORIES:
            img_dir = paths.data_root / cat / cat / "image"
            if img_dir.is_dir():
                total += sum(1 for f in img_dir.iterdir() if f.is_file())
        summary["images"] = total

    csv_path = paths.annotations_csv
    if csv_path.is_file():
        annotations = load_annotations_csv(csv_path)
        summary["annotated_images"] = len(annotations)
        summary["total_boxes"] = sum(len(v) for v in annotations.values())
        summary["liver_boxes"] = sum(
            sum(1 for b in v if int(b["class_id"]) == 0) for v in annotations.values()
        )
        summary["mass_boxes"] = sum(
            sum(1 for b in v if int(b["class_id"]) == 1) for v in annotations.values()
        )
    else:
        summary["annotated_images"] = 0
        summary["total_boxes"] = 0

    summary["classes"] = ", ".join(CLASS_NAMES)
    return summary


# ---------------------------------------------------------------------------
# Synthetic / demo data for development
# ---------------------------------------------------------------------------

def create_synthetic_liver_dataset(dest_dir: Path, *, n_samples: int = 30) -> LiverDatasetPaths:
    """Generate a tiny synthetic dataset for smoke-testing the YOLO pipeline.

    Creates random grayscale "ultrasound-like" images with random bounding
    boxes so the full training loop can be exercised without real data.
    """

    dest_dir = Path(dest_dir)
    flat_dir = dest_dir / "images_flat"
    flat_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    rows: list[dict[str, str]] = []

    for i in range(n_samples):
        w, h = rng.randint(256, 513), rng.randint(256, 513)
        base = rng.randint(20, 60, size=(h, w), dtype=np.uint8)
        noise = rng.normal(0, 15, size=(h, w)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Liver region
        x1 = rng.randint(10, w // 2)
        y1 = rng.randint(10, h // 2)
        x2 = rng.randint(w // 2 + 10, w - 5)
        y2 = rng.randint(h // 2 + 10, h - 5)
        img_arr[y1:y2, x1:x2] = np.clip(
            img_arr[y1:y2, x1:x2].astype(np.int16) + rng.randint(40, 80), 0, 255,
        ).astype(np.uint8)

        image_id = f"synth_{i:04d}"
        Image.fromarray(img_arr, mode="L").convert("RGB").save(flat_dir / f"{image_id}.png")

        rows.append({
            "image_id": image_id,
            "x_min": str(x1), "y_min": str(y1),
            "x_max": str(x2), "y_max": str(y2),
            "class_id": "0", "category": "Synthetic",
        })

    csv_path = dest_dir / "annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max", "class_id", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Created synthetic liver dataset: %d samples at %s", n_samples, dest_dir)
    return LiverDatasetPaths(root=dest_dir)
