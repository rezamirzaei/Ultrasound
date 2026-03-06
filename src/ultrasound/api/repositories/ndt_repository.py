"""NDT-specific repository operations."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from ultrasound.api.database.models import NdtDefectORM, NdtSampleORM
from ultrasound.api.models.domain import NdtDefectRecord, NdtSampleRecord
from ultrasound.api.repositories.dataset_support import DatasetRepositorySupport


class NdtRepository(DatasetRepositorySupport):
    """Persist and read NDT waveform samples and metadata."""

    def sync_ndt_from_filesystem(self) -> int:
        fingerprint = self._compute_ndt_fingerprint()
        if self._meta_get("ndt_fingerprint") == fingerprint:
            return 0

        inserted = 0
        with self.db.session_scope() as session:
            session.execute(delete(NdtDefectORM))
            session.execute(delete(NdtSampleORM))
            if self.config.ndt_dir.exists():
                for sample_path in sorted(self.config.ndt_dir.glob("*.npz")):
                    data = np.load(sample_path, allow_pickle=True)
                    rf = np.asarray(data["rf"], dtype=np.float64).reshape(-1)
                    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
                    fs_hz = self._to_float_scalar(data.get("fs", 50e6), 50e6)
                    fc_hz = self._to_float_scalar(data.get("fc", 5e6), 5e6)
                    c_mps = self._to_float_scalar(data.get("c", 5900.0), 5900.0)
                    thickness_raw = self._to_float_scalar(data.get("thickness", np.nan), np.nan)
                    thickness_m = thickness_raw if np.isfinite(thickness_raw) and thickness_raw > 0 else None
                    description = str(data.get("description", sample_path.name))
                    defects = self._build_defect_records(data.get("defects", np.array([], dtype=object)))
                    source_hash = hashlib.sha256(rf.tobytes() + time.tobytes() + description.encode("utf-8")).hexdigest()
                    sample = NdtSampleORM(
                        name=sample_path.name,
                        rf_blob=self._array_to_blob(rf),
                        time_blob=self._array_to_blob(time),
                        n_points=int(rf.size),
                        fs_hz=float(fs_hz),
                        fc_hz=float(fc_hz),
                        c_mps=float(c_mps),
                        thickness_m=thickness_m,
                        description=description,
                        source_hash=source_hash,
                    )
                    sample.defects = [
                        NdtDefectORM(ordinal=i, depth_m=defect.depth_m, amplitude=defect.amplitude)
                        for i, defect in enumerate(defects)
                    ]
                    session.add(sample)
                    inserted += 1
            self._set_meta_value(session, "ndt_fingerprint", fingerprint)
        return inserted

    def list_ndt_samples(self) -> list[str]:
        with self.db.session_scope() as session:
            names = session.scalars(select(NdtSampleORM.name).order_by(NdtSampleORM.name)).all()
        return [str(name) for name in names]

    def load_ndt_sample(self, sample_name: str) -> NdtSampleRecord:
        with self.db.session_scope() as session:
            sample = session.scalars(
                select(NdtSampleORM)
                .options(selectinload(NdtSampleORM.defects))
                .where(NdtSampleORM.name == sample_name)
                .limit(1)
            ).first()
        if sample is None:
            available = self.list_ndt_samples()
            raise FileNotFoundError(f"Missing NDT sample '{sample_name}'. Available: {available}")
        return NdtSampleRecord(
            name=sample.name,
            path=self.config.ndt_dir / sample.name,
            rf=self._blob_to_array(sample.rf_blob),
            time=self._blob_to_array(sample.time_blob),
            fs_hz=float(sample.fs_hz),
            fc_hz=float(sample.fc_hz),
            c_mps=float(sample.c_mps),
            thickness_m=float(sample.thickness_m) if sample.thickness_m is not None else None,
            description=sample.description,
            defects=[
                NdtDefectRecord(depth_m=defect.depth_m, amplitude=defect.amplitude)
                for defect in sorted(sample.defects, key=lambda item: item.ordinal)
            ],
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
                        {"depth_m": defect.depth_m, "amplitude": defect.amplitude}
                        for defect in sample.defects
                    ],
                }
            )
        return rows
