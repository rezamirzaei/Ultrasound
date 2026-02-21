"""Repository layer for SQL-backed BUSI and filesystem-backed NDT access."""

from __future__ import annotations

from typing import Any

import numpy as np

from ultrasound.api.config import AppConfig
from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
    NdtDefectRecord,
    NdtSampleRecord,
)
from ultrasound.api.repositories.busi_sql_repository import BusiSqlRepository


class DatasetRepository:
    """Encapsulates dataset access and metadata extraction."""

    CLASSES = ("benign", "malignant", "normal")

    def __init__(self, config: AppConfig):
        self.config = config
        self.busi_sql_repository = BusiSqlRepository(
            db_path=self.config.data_dir / "inphase.sqlite3",
            busi_dir=self.config.busi_dir,
        )

    def get_busi_counts(self) -> dict[str, int]:
        return self.busi_sql_repository.get_busi_counts()

    def get_busi_sample(self, class_name: str, index: int = 0) -> BusiSampleRecord:
        return self.busi_sql_repository.get_busi_sample(class_name=class_name, index=index)

    def list_busi_training_samples(
        self, include_normal: bool = False
    ) -> list[BusiTrainingSampleRecord]:
        return self.busi_sql_repository.list_busi_training_samples(include_normal=include_normal)

    def save_busi_training_run(self, run: BusiTrainingRunRecord) -> BusiTrainingRunRecord:
        return self.busi_sql_repository.save_busi_training_run(run)

    def get_latest_busi_training_run(
        self, include_normal: bool = False
    ) -> BusiTrainingRunRecord | None:
        return self.busi_sql_repository.get_latest_busi_training_run(include_normal=include_normal)

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
        """Parse defect data from numpy files.

        Handles multiple storage formats:
        - 2D float array of shape (N, 2): rows are [depth_m, amplitude]
        - List of dicts with 'depth_m' and 'amplitude' keys
        - List of tuples/lists of (depth_m, amplitude)
        - Object arrays with mixed content
        """
        arr = np.asarray(defects_obj)

        # Fast path: 2D numeric array with shape (N, 2)
        if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
            records: list[NdtDefectRecord] = []
            for row in arr:
                depth = float(row[0]) if np.isfinite(row[0]) else None
                amp = float(row[1]) if np.isfinite(row[1]) else None
                if depth is not None or amp is not None:
                    records.append(NdtDefectRecord(depth_m=depth, amplitude=amp))
            return records

        # Empty array
        if arr.size == 0:
            return []

        # General case: convert to Python list and iterate
        try:
            defects_raw = arr.tolist()
            if not isinstance(defects_raw, list):
                defects_raw = [defects_raw]
        except Exception:
            return []

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
                continue

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
