"""Repository layer for filesystem-backed dataset access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.models.domain import BusiSampleRecord, NdtDefectRecord, NdtSampleRecord


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

    def get_busi_sample(self, class_name: str, index: int = 0) -> BusiSampleRecord:
        class_dir = self.config.busi_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"BUSI class directory not found: {class_dir}")

        images = sorted(p for p in class_dir.glob("*.png") if "_mask" not in p.stem)
        if not images:
            raise FileNotFoundError(f"No BUSI images found in {class_dir}")

        resolved_index = int(index % len(images))
        image_path = images[resolved_index]
        mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if mask_candidates:
            mask = np.asarray(Image.open(mask_candidates[0]).convert("L"), dtype=np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return BusiSampleRecord(
            class_name=class_name,
            requested_index=index,
            resolved_index=resolved_index,
            total_samples=len(images),
            image_path=image_path,
            image_rgb=image,
            mask=mask,
        )

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

    def _build_defect_records(self, defects_obj: Any) -> list[NdtDefectRecord]:
        try:
            defects_raw = np.asarray(defects_obj, dtype=object).tolist()
            if not isinstance(defects_raw, list):
                defects_raw = [defects_raw]
        except Exception:
            defects_raw = []

        defects: list[NdtDefectRecord] = []
        for item in defects_raw:
            if isinstance(item, dict):
                record = NdtDefectRecord(
                    depth_m=item.get("depth_m"),
                    amplitude=item.get("amplitude"),
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                record = NdtDefectRecord(
                    depth_m=item[0],
                    amplitude=item[1],
                )
            else:
                record = NdtDefectRecord()

            # Keep only entries that carry at least one finite scalar.
            if record.depth_m is not None or record.amplitude is not None:
                defects.append(record)
        return defects

    def load_ndt_sample(self, sample_name: str) -> NdtSampleRecord:
        sample_path = self.config.ndt_dir / sample_name
        if not sample_path.exists():
            available = self.list_ndt_samples()
            raise FileNotFoundError(f"Missing NDT sample '{sample_name}'. Available: {available}")

        data = np.load(sample_path, allow_pickle=True)
        defects = self._build_defect_records(data.get("defects", np.array([], dtype=object)))

        return NdtSampleRecord(
            name=sample_name,
            path=sample_path,
            rf=np.asarray(data["rf"], dtype=np.float64).reshape(-1),
            time=np.asarray(data["time"], dtype=np.float64).reshape(-1),
            fs_hz=self._to_float_scalar(data.get("fs", 50e6), 50e6),
            fc_hz=self._to_float_scalar(data.get("fc", 5e6), 5e6),
            c_mps=self._to_float_scalar(data.get("c", 5900.0), 5900.0),
            thickness_m=self._to_float_scalar(data.get("thickness", np.nan), np.nan),
            description=str(data.get("description", sample_name)),
            defects=defects,
        )

    def summarize_ndt_samples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.list_ndt_samples():
            sample = self.load_ndt_sample(name)
            rows.append(
                {
                    "name": sample.name,
                    "n_points": int(sample.rf.size),
                    "fs_hz": float(sample.fs_hz),
                    "fc_hz": float(sample.fc_hz),
                    "thickness_mm": float(sample.thickness_m * 1e3) if sample.thickness_m else None,
                    "n_defects": len(sample.defects),
                    "description": sample.description,
                    "defects": [
                        {
                            "depth_m": defect.depth_m,
                            "amplitude": defect.amplitude,
                        }
                        for defect in sample.defects
                    ],
                }
            )
        return rows
