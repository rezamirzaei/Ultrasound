"""Repository layer for filesystem-backed dataset access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultrasound.api.config import AppConfig


class DatasetRepository:
    """Encapsulates raw dataset access and metadata extraction."""

    CLASSES = ("benign", "malignant", "normal")

    def __init__(self, config: AppConfig):
        self.config = config

    def get_busi_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for class_name in self.CLASSES:
            class_dir = self.config.busi_dir / class_name
            if not class_dir.exists():
                counts[class_name] = 0
                continue
            counts[class_name] = len([p for p in class_dir.glob("*.png") if "_mask" not in p.stem])
        return counts

    def get_busi_sample(self, class_name: str, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        class_dir = self.config.busi_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"BUSI class directory not found: {class_dir}")

        images = sorted(p for p in class_dir.glob("*.png") if "_mask" not in p.stem)
        if not images:
            raise FileNotFoundError(f"No BUSI images found in {class_dir}")

        image_path = images[index % len(images)]
        mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if mask_candidates:
            mask = np.asarray(Image.open(mask_candidates[0]).convert("L"), dtype=np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return image, mask

    def list_ndt_samples(self) -> list[str]:
        if not self.config.ndt_dir.exists():
            return []
        return sorted(path.name for path in self.config.ndt_dir.glob("*.npz"))

    def _to_float_scalar(self, value: Any, default: float) -> float:
        try:
            arr = np.asarray(value)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    def load_ndt_sample(self, sample_name: str) -> dict[str, Any]:
        sample_path = self.config.ndt_dir / sample_name
        if not sample_path.exists():
            available = self.list_ndt_samples()
            raise FileNotFoundError(f"Missing NDT sample '{sample_name}'. Available: {available}")

        data = np.load(sample_path, allow_pickle=True)
        rf = np.asarray(data["rf"], dtype=np.float64).reshape(-1)
        time = np.asarray(data["time"], dtype=np.float64).reshape(-1)

        defects_obj = data.get("defects", np.array([], dtype=object))
        try:
            defects_raw = defects_obj.tolist()
            if not isinstance(defects_raw, list):
                defects_raw = [defects_raw]
        except Exception:
            defects_raw = []

        defects: list[dict[str, float]] = []
        for item in defects_raw:
            if isinstance(item, dict):
                depth = self._to_float_scalar(item.get("depth_m", np.nan), np.nan)
                amp = self._to_float_scalar(item.get("amplitude", np.nan), np.nan)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                depth = self._to_float_scalar(item[0], np.nan)
                amp = self._to_float_scalar(item[1], np.nan)
            else:
                continue
            defects.append({"depth_m": depth, "amplitude": amp})

        return {
            "name": sample_name,
            "path": str(sample_path),
            "rf": rf,
            "time": time,
            "fs": self._to_float_scalar(data.get("fs", 50e6), 50e6),
            "fc": self._to_float_scalar(data.get("fc", 5e6), 5e6),
            "c": self._to_float_scalar(data.get("c", 5900.0), 5900.0),
            "thickness": self._to_float_scalar(data.get("thickness", np.nan), np.nan),
            "description": str(data.get("description", sample_name)),
            "defects": defects,
        }

    def summarize_ndt_samples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.list_ndt_samples():
            sample = self.load_ndt_sample(name)
            rows.append(
                {
                    "name": sample["name"],
                    "n_points": int(sample["rf"].size),
                    "fs_hz": float(sample["fs"]),
                    "fc_hz": float(sample["fc"]),
                    "thickness_mm": float(sample["thickness"] * 1e3),
                    "n_defects": len(sample["defects"]),
                    "description": sample["description"],
                    "defects": sample["defects"],
                }
            )
        return rows
