"""Liver Ultrasound Detection dataset utilities.

Handles downloading, parsing, and preparing the Kaggle
*liver-ultrasound-detection* competition dataset for YOLO training.

Competition: https://www.kaggle.com/competitions/liver-ultrasound-detection/data

The dataset provides:
  - ``train/`` images  (ultrasound scans)
  - ``train.csv``      (bounding-box annotations: image_id, x_min, y_min, x_max, y_max)
  - ``test/`` images   (no labels — competition hold-out)
"""

from __future__ import annotations

import csv
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("inphase.yolo.liver")

# The competition has a single detection class: "liver".
CLASS_NAMES: list[str] = ["liver"]


@dataclass(frozen=True)
class LiverDatasetPaths:
    """Resolved paths for the liver ultrasound detection dataset."""

    root: Path
    train_images_dir: Path
    test_images_dir: Path
    annotations_csv: Path

    @property
    def is_ready(self) -> bool:
        return (
            self.train_images_dir.is_dir()
            and self.annotations_csv.is_file()
            and any(self.train_images_dir.iterdir())
        )


def resolve_liver_paths(data_dir: Path) -> LiverDatasetPaths:
    """Build expected paths under ``data_dir/liver_ultrasound_detection/``."""
    root = data_dir / "liver_ultrasound_detection"
    return LiverDatasetPaths(
        root=root,
        train_images_dir=root / "train",
        test_images_dir=root / "test",
        annotations_csv=root / "train.csv",
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_liver_dataset(dest_dir: Path, *, force: bool = False) -> LiverDatasetPaths:
    """Download the competition data from Kaggle (requires credentials).

    Credentials: ``~/.kaggle/kaggle.json`` or env vars ``KAGGLE_USERNAME``
    / ``KAGGLE_KEY``.

    If a ZIP file is already present in *dest_dir*, it will be extracted
    without re-downloading (unless *force* is ``True``).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = LiverDatasetPaths(
        root=dest_dir,
        train_images_dir=dest_dir / "train",
        test_images_dir=dest_dir / "test",
        annotations_csv=dest_dir / "train.csv",
    )

    # Fast path: already extracted.
    if paths.is_ready and not force:
        logger.info("Dataset already present at %s", dest_dir)
        return paths

    # Check for a pre-placed ZIP.
    zip_candidates = sorted(dest_dir.glob("*.zip"))
    if zip_candidates and not force:
        _extract_zip(zip_candidates[0], dest_dir)
        return paths

    _download_from_kaggle(dest_dir, force=force)
    return paths


def _download_from_kaggle(dest_dir: Path, *, force: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Kaggle API not installed. Run: pip install kaggle\n"
            "Then set up credentials: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication"
        ) from exc

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading liver-ultrasound-detection from Kaggle …")
    api.competition_download_files(
        "liver-ultrasound-detection",
        path=str(dest_dir),
        force=bool(force),
        quiet=False,
    )

    for zp in sorted(dest_dir.glob("*.zip")):
        _extract_zip(zp, dest_dir)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s → %s", zip_path.name, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotations_csv(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    """Parse ``train.csv`` → ``{image_id: [{x_min, y_min, x_max, y_max, class_id}]}``."""
    annotations: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row["image_id"].strip()
            annotations.setdefault(image_id, []).append({
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
                "class_id": 0,  # single-class: liver
            })
    return annotations


def summarize_dataset(paths: LiverDatasetPaths) -> dict[str, int | str]:
    """Quick summary of the dataset on disk."""
    summary: dict[str, int | str] = {"root": str(paths.root)}

    if paths.train_images_dir.is_dir():
        train_images = list(paths.train_images_dir.iterdir())
        summary["train_images"] = len([f for f in train_images if f.is_file()])
    else:
        summary["train_images"] = 0

    if paths.test_images_dir.is_dir():
        test_images = list(paths.test_images_dir.iterdir())
        summary["test_images"] = len([f for f in test_images if f.is_file()])
    else:
        summary["test_images"] = 0

    if paths.annotations_csv.is_file():
        annotations = load_annotations_csv(paths.annotations_csv)
        summary["annotated_images"] = len(annotations)
        summary["total_boxes"] = sum(len(v) for v in annotations.values())
    else:
        summary["annotated_images"] = 0
        summary["total_boxes"] = 0

    return summary


# ---------------------------------------------------------------------------
# Synthetic / demo data for development without Kaggle credentials
# ---------------------------------------------------------------------------

def create_synthetic_liver_dataset(dest_dir: Path, *, n_samples: int = 30) -> LiverDatasetPaths:
    """Generate a tiny synthetic dataset for smoke-testing the YOLO pipeline.

    Creates random grayscale "ultrasound-like" images with random bounding
    boxes so the full training loop can be exercised without real data.
    """
    import numpy as np
    from PIL import Image

    dest_dir = Path(dest_dir)
    train_dir = dest_dir / "train"
    test_dir = dest_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    rows: list[dict[str, str]] = []

    for i in range(n_samples):
        w, h = rng.randint(256, 513), rng.randint(256, 513)
        # Simulate ultrasound-like texture: dark with speckle noise.
        base = rng.randint(20, 60, size=(h, w), dtype=np.uint8)
        noise = rng.normal(0, 15, size=(h, w)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a brighter "liver" region.
        x1 = rng.randint(10, w // 2)
        y1 = rng.randint(10, h // 2)
        x2 = rng.randint(w // 2 + 10, w - 5)
        y2 = rng.randint(h // 2 + 10, h - 5)
        img_arr[y1:y2, x1:x2] = np.clip(
            img_arr[y1:y2, x1:x2].astype(np.int16) + rng.randint(40, 80),
            0, 255,
        ).astype(np.uint8)

        image_id = f"synth_{i:04d}"
        img_path = train_dir / f"{image_id}.png"
        Image.fromarray(img_arr, mode="L").convert("RGB").save(img_path)

        rows.append({
            "image_id": image_id,
            "x_min": str(x1),
            "y_min": str(y1),
            "x_max": str(x2),
            "y_max": str(y2),
        })

    # A few test images (no labels).
    for i in range(5):
        w, h = 320, 320
        img_arr = rng.randint(20, 80, size=(h, w), dtype=np.uint8)
        img_path = test_dir / f"synth_test_{i:04d}.png"
        Image.fromarray(img_arr, mode="L").convert("RGB").save(img_path)

    csv_path = dest_dir / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Created synthetic liver dataset: %d train, 5 test at %s", n_samples, dest_dir)

    return LiverDatasetPaths(
        root=dest_dir,
        train_images_dir=train_dir,
        test_images_dir=test_dir,
        annotations_csv=csv_path,
    )

