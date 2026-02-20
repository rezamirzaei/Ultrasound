"""Shared utilities for reproducible notebook execution."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

NDT_SAMPLE_DIR = Path("data/ascan_signals/ndt_samples")
BUSI_DIR = Path("data/busi")
NOTEBOOK_OUTPUT_ROOT = Path("outputs/notebooks")


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve repository root from either project root or notebooks/ cwd."""
    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml and src/")


def ensure_src_on_path() -> Path:
    """Add src directory to sys.path and return project root."""
    root = resolve_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def set_reproducible_seed(seed: int = 42) -> int:
    """Set deterministic seeds for numpy and random."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def ensure_notebook_output_dir(notebook_name: str) -> Path:
    """Create and return a per-notebook output directory."""
    root = resolve_project_root()
    output_dir = root / NOTEBOOK_OUTPUT_ROOT / notebook_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def busi_class_counts(data_dir: Path | None = None) -> dict[str, int]:
    """Return BUSI class image counts excluding mask files."""
    root = resolve_project_root()
    base = root / (data_dir or BUSI_DIR)
    counts: dict[str, int] = {}
    for class_name in ("benign", "malignant", "normal"):
        class_dir = base / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue
        counts[class_name] = len([p for p in class_dir.glob("*.png") if "_mask" not in p.stem])
    return counts


def find_busi_sample(
    class_name: str = "benign",
    data_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """Locate one raw BUSI image and its first mask (if present)."""
    root = resolve_project_root()
    base = root / (data_dir or BUSI_DIR)
    class_dir = base / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"BUSI class directory not found: {class_dir}")

    images = sorted(p for p in class_dir.glob("*.png") if "_mask" not in p.stem)
    if not images:
        raise FileNotFoundError(f"No images found under {class_dir}")

    image_path = images[0]
    masks = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
    return image_path, (masks[0] if masks else None)


def load_busi_sample_arrays(
    class_name: str = "benign",
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one BUSI image/mask pair as numpy arrays."""
    image_path, mask_path = find_busi_sample(class_name=class_name, data_dir=data_dir)
    image = np.array(Image.open(image_path).convert("RGB"))
    if mask_path is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask = np.array(Image.open(mask_path).convert("L"))
    return image, mask


def to_float_scalar(value: Any, default: float) -> float:
    try:
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def load_ndt_sample(sample_name: str = "weld_inspection.npz") -> Dict[str, Any]:
    """Load a local NDT A-scan sample from data/ascan_signals/ndt_samples."""
    root = resolve_project_root()
    sample_path = root / NDT_SAMPLE_DIR / sample_name
    if not sample_path.exists():
        available = sorted(p.name for p in (root / NDT_SAMPLE_DIR).glob("*.npz"))
        raise FileNotFoundError(f"Missing sample: {sample_path}. Available samples: {available}")

    data = np.load(sample_path, allow_pickle=True)
    rf = np.asarray(data["rf"], dtype=np.float64).reshape(-1)
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)

    # Make metadata robust to both python scalars and numpy arrays.
    fs = to_float_scalar(data["fs"], default=50e6)
    fc = to_float_scalar(data["fc"], default=5e6)
    c = to_float_scalar(data["c"], default=5900.0)
    thickness = to_float_scalar(data.get("thickness", np.nan), default=np.nan)

    description = str(data.get("description", sample_name))
    defects_obj = data.get("defects", np.array([], dtype=object))
    defects_list: list[Any]
    try:
        defects_list = list(defects_obj.tolist())
    except Exception:
        defects_list = []

    return {
        "path": str(sample_path),
        "name": sample_name,
        "rf": rf,
        "time": time,
        "fs": fs,
        "fc": fc,
        "c": c,
        "thickness": thickness,
        "description": description,
        "defects": defects_list,
    }


def summarize_ndt_samples() -> list[dict[str, Any]]:
    """Return metadata summary for all local NDT samples."""
    root = resolve_project_root()
    rows: list[dict[str, Any]] = []
    for npz in sorted((root / NDT_SAMPLE_DIR).glob("*.npz")):
        sample = load_ndt_sample(npz.name)
        rows.append(
            {
                "sample": sample["name"],
                "n_points": int(sample["rf"].size),
                "fs_hz": sample["fs"],
                "fc_hz": sample["fc"],
                "thickness_m": sample["thickness"],
                "n_defects": len(sample["defects"]),
            }
        )
    return rows


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Save report payload as JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
