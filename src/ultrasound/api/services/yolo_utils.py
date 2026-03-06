"""Shared helpers for YOLO workflows (labels, boxes, and lightweight asset downloads)."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ultrasound.api.models.schemas import (
    DownloadedAssetManifest,
    YoloLabel,
    YoloXyxyBox,
)

# ---------------------------------------------------------------------------
# Asset manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> DownloadedAssetManifest | None:
    if not path.exists():
        return None
    try:
        return DownloadedAssetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError):
        return None


def write_manifest(path: Path, manifest: DownloadedAssetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# File download with checksum
# ---------------------------------------------------------------------------

def download_url_to_path(
    url: str,
    dest_path: Path,
    *,
    timeout_seconds: int = 180,
    chunk_bytes: int = 1024 * 1024,
    user_agent: str = "inPhase-ultrasound-imaging-toolkit/1.0",
) -> DownloadedAssetManifest:
    """Download *url* to *dest_path* and return a manifest with SHA-256."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    sha = hashlib.sha256()
    size_bytes = 0
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response, \
                tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(int(chunk_bytes))
                if not chunk:
                    break
                handle.write(chunk)
                sha.update(chunk)
                size_bytes += len(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    try:
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return DownloadedAssetManifest(
        source_url=url,
        downloaded_at=datetime.now(tz=timezone.utc),
        sha256=sha.hexdigest(),
        size_bytes=int(size_bytes),
    )


# ---------------------------------------------------------------------------
# Bounding-box / label conversions
# ---------------------------------------------------------------------------

def mask_to_xyxy(mask: np.ndarray) -> YoloXyxyBox | None:
    """Compute a tight bounding box over non-zero pixels in a 2-D mask."""
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim != 2:
        raise ValueError("Expected mask to be single-channel [H, W]")
    points = np.argwhere(mask_arr > 0)
    if points.size == 0:
        return None
    y1, x1 = int(points[:, 0].min()), int(points[:, 1].min())
    y2, x2 = int(points[:, 0].max()), int(points[:, 1].max())
    return YoloXyxyBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def xyxy_to_yolo_label(
    *,
    bbox: YoloXyxyBox,
    class_id: int,
    class_name: str | None,
    image_width: int,
    image_height: int,
) -> YoloLabel:
    """Convert pixel xyxy box into a normalized YOLO xywh label."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width / image_height must be positive")

    x1, y1 = float(bbox.x1), float(bbox.y1)
    x2, y2 = float(bbox.x2), float(bbox.y2)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox coordinates must satisfy x2 >= x1 and y2 >= y1")

    max_x = float(image_width - 1)
    max_y = float(image_height - 1)
    x1 = min(max(x1, 0.0), max_x)
    y1 = min(max(y1, 0.0), max_y)
    x2 = min(max(x2, 0.0), max_x)
    y2 = min(max(y2, 0.0), max_y)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox does not intersect image bounds")

    box_w = max(1.0, x2 - x1 + 1.0)
    box_h = max(1.0, y2 - y1 + 1.0)
    x_center = (x1 + x2 + 1.0) / 2.0
    y_center = (y1 + y2 + 1.0) / 2.0

    return YoloLabel(
        class_id=int(class_id),
        class_name=class_name,
        x_center=float(x_center / float(image_width)),
        y_center=float(y_center / float(image_height)),
        width=float(box_w / float(image_width)),
        height=float(box_h / float(image_height)),
    )


# ---------------------------------------------------------------------------
# YOLO .txt label formatting / parsing
# ---------------------------------------------------------------------------

def format_yolo_labels(labels: list[YoloLabel]) -> str:
    """Render a list of labels as a YOLO ``.txt`` string."""
    if not labels:
        return ""
    lines = [
        f"{lb.class_id} {lb.x_center:.6f} {lb.y_center:.6f} {lb.width:.6f} {lb.height:.6f}"
        for lb in labels
    ]
    return "\n".join(lines) + "\n"


def parse_yolo_txt_labels(
    labels_text: str,
    *,
    class_names: list[str] | None = None,
) -> list[YoloLabel]:
    """Parse a YOLO .txt payload into validated label rows."""
    safe_class_names = class_names or []

    parsed: list[YoloLabel] = []
    for line_number, raw in enumerate((labels_text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label on line {line_number}: expected 5 columns")
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except Exception as exc:
            raise ValueError(f"Invalid numeric values on line {line_number}") from exc

        if class_id < 0:
            raise ValueError(f"class_id {class_id} must be non-negative on line {line_number}")
        if safe_class_names and class_id >= len(safe_class_names):
            raise ValueError(
                f"class_id {class_id} out of range (0..{len(safe_class_names) - 1}) on line {line_number}"
            )

        parsed.append(
            YoloLabel(
                class_id=class_id,
                class_name=safe_class_names[class_id] if class_id < len(safe_class_names) else None,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return parsed
