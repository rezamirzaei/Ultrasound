"""Repository layer for SQLAlchemy-backed dataset access."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from ultrasound.api.config import AppConfig
from ultrasound.api.database.models import (
    BusiSampleORM,
    BusiTrainingRunORM,
    DatasetMetaORM,
    NdtDefectORM,
    NdtSampleORM,
)
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
    NdtDefectRecord,
    NdtSampleRecord,
)


class DatasetRepository:
    """Encapsulates dataset access and metadata extraction."""

    CLASSES = ("benign", "malignant", "normal")
    CLASS_TO_LABEL = {"benign": 0, "malignant": 1, "normal": 2}

    def __init__(self, config: AppConfig, db: DatabaseSessionManager):
        self.config = config
        self.db = db
        self.db.create_schema()
        self.sync_from_sources()

    def _to_float_scalar(self, value: Any, default: float) -> float:
        try:
            arr = np.asarray(value)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    def _array_to_blob(self, arr: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, np.asarray(arr, dtype=np.float64), allow_pickle=False)
        return buffer.getvalue()

    def _blob_to_array(self, blob: bytes) -> np.ndarray:
        buffer = BytesIO(blob)
        arr = np.load(buffer, allow_pickle=False)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    def _canonical_png_rgb(self, image_path: Path) -> tuple[bytes, int, int]:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
            width, height = image_rgb.size
            buffer = BytesIO()
            image_rgb.save(buffer, format="PNG")
        return buffer.getvalue(), int(width), int(height)

    def _canonical_png_mask(self, mask_path: Path | None, width: int, height: int) -> bytes | None:
        if mask_path is None:
            return None
        with Image.open(mask_path) as pil_mask:
            mask_gray = pil_mask.convert("L")
            if mask_gray.size != (width, height):
                mask_gray = mask_gray.resize((width, height), Image.Resampling.NEAREST)
            buffer = BytesIO()
            mask_gray.save(buffer, format="PNG")
        return buffer.getvalue()

    def _decode_rgb_blob(self, blob: bytes) -> np.ndarray:
        with Image.open(BytesIO(blob)) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        return image

    def _decode_mask_blob(self, blob: bytes | None, shape: tuple[int, int]) -> np.ndarray:
        if blob is None:
            return np.zeros(shape, dtype=np.uint8)
        with Image.open(BytesIO(blob)) as pil_mask:
            mask = np.asarray(pil_mask.convert("L"), dtype=np.uint8)
        if mask.shape != shape:
            resized = Image.fromarray(mask, mode="L").resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
            mask = np.asarray(resized, dtype=np.uint8)
        return mask

    def _resolve_train_cutoff(self, n_samples: int) -> int:
        if n_samples <= 1:
            return n_samples
        train_count = int(round(0.8 * float(n_samples)))
        train_count = max(1, min(train_count, n_samples - 1))
        return train_count

    def _meta_get(self, session_key: str) -> str | None:
        with self.db.session_scope() as session:
            row = session.get(DatasetMetaORM, session_key)
            return row.value if row is not None else None

    def _meta_upsert(self, key: str, value: str) -> None:
        with self.db.session_scope() as session:
            row = session.get(DatasetMetaORM, key)
            if row is None:
                session.add(DatasetMetaORM(key=key, value=value))
            else:
                row.value = value

    def _compute_busi_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for class_name in self.CLASSES:
            class_dir = self.config.busi_dir / class_name
            if not class_dir.exists():
                digest.update(f"{class_name}:missing|".encode("utf-8"))
                continue

            image_paths = sorted(
                path for path in class_dir.glob("*.png") if "_mask" not in path.stem
            )
            digest.update(f"{class_name}:{len(image_paths)}|".encode("utf-8"))
            for image_path in image_paths:
                stat = image_path.stat()
                digest.update(
                    f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode("utf-8")
                )
                mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                if mask_candidates:
                    mask_stat = mask_candidates[0].stat()
                    digest.update(
                        f"{mask_candidates[0].name}:{mask_stat.st_size}:{mask_stat.st_mtime_ns}|".encode(
                            "utf-8"
                        )
                    )
        return digest.hexdigest()

    def _compute_ndt_fingerprint(self) -> str:
        digest = hashlib.sha256()
        if not self.config.ndt_dir.exists():
            digest.update(b"ndt:missing")
            return digest.hexdigest()

        sample_paths = sorted(self.config.ndt_dir.glob("*.npz"))
        digest.update(f"ndt:{len(sample_paths)}|".encode("utf-8"))
        for sample_path in sample_paths:
            stat = sample_path.stat()
            digest.update(f"{sample_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode("utf-8"))
        return digest.hexdigest()

    def _build_defect_records(self, defects_obj: Any) -> list[NdtDefectRecord]:
        """Parse defect data from numpy files."""
        arr = np.asarray(defects_obj)

        if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
            records: list[NdtDefectRecord] = []
            for row in arr:
                depth = float(row[0]) if np.isfinite(row[0]) else None
                amp = float(row[1]) if np.isfinite(row[1]) else None
                if depth is not None or amp is not None:
                    records.append(NdtDefectRecord(depth_m=depth, amplitude=amp))
            return records

        if arr.size == 0:
            return []

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
                record = NdtDefectRecord(depth_m=item[0], amplitude=item[1])
            else:
                continue

            if record.depth_m is not None or record.amplitude is not None:
                defects.append(record)
        return defects

    def sync_busi_from_filesystem(self) -> int:
        fingerprint = self._compute_busi_fingerprint()
        if self._meta_get("busi_fingerprint") == fingerprint:
            return 0

        inserted = 0
        with self.db.session_scope() as session:
            session.execute(delete(BusiSampleORM))

            for class_name in self.CLASSES:
                class_dir = self.config.busi_dir / class_name
                if not class_dir.exists():
                    continue

                image_paths = sorted(
                    path for path in class_dir.glob("*.png") if "_mask" not in path.stem
                )
                train_cutoff = self._resolve_train_cutoff(len(image_paths))

                for index, image_path in enumerate(image_paths):
                    image_blob, width, height = self._canonical_png_rgb(image_path)
                    mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                    mask_blob = self._canonical_png_mask(
                        mask_candidates[0] if mask_candidates else None,
                        width=width,
                        height=height,
                    )
                    source_hash = hashlib.sha256(image_blob + (mask_blob or b"")).hexdigest()
                    split = "train" if index < train_cutoff else "test"
                    session.add(
                        BusiSampleORM(
                            class_name=class_name,
                            image_filename=image_path.name,
                            sample_stem=image_path.stem,
                            image_blob=image_blob,
                            mask_blob=mask_blob,
                            width=width,
                            height=height,
                            label=self.CLASS_TO_LABEL[class_name],
                            split=split,
                            source_hash=source_hash,
                        )
                    )
                    inserted += 1

            row = session.get(DatasetMetaORM, "busi_fingerprint")
            if row is None:
                session.add(DatasetMetaORM(key="busi_fingerprint", value=fingerprint))
            else:
                row.value = fingerprint

        return inserted

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
                    thickness_m = (
                        thickness_raw if np.isfinite(thickness_raw) and thickness_raw > 0 else None
                    )
                    description = str(data.get("description", sample_path.name))

                    defects = self._build_defect_records(
                        data.get("defects", np.array([], dtype=object))
                    )
                    source_hash = hashlib.sha256(
                        rf.tobytes() + time.tobytes() + description.encode("utf-8")
                    ).hexdigest()

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
                        NdtDefectORM(
                            ordinal=i,
                            depth_m=defect.depth_m,
                            amplitude=defect.amplitude,
                        )
                        for i, defect in enumerate(defects)
                    ]
                    session.add(sample)
                    inserted += 1

            row = session.get(DatasetMetaORM, "ndt_fingerprint")
            if row is None:
                session.add(DatasetMetaORM(key="ndt_fingerprint", value=fingerprint))
            else:
                row.value = fingerprint

        return inserted

    def sync_from_sources(self) -> None:
        self.sync_busi_from_filesystem()
        self.sync_ndt_from_filesystem()

    def get_busi_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.CLASSES}
        with self.db.session_scope() as session:
            rows = session.execute(
                select(BusiSampleORM.class_name, func.count(BusiSampleORM.id)).group_by(
                    BusiSampleORM.class_name
                )
            ).all()
            for class_name, n in rows:
                if class_name in counts:
                    counts[class_name] = int(n)
        return counts

    def get_busi_sample(self, class_name: str, index: int = 0) -> BusiSampleRecord:
        if class_name not in self.CLASSES:
            raise FileNotFoundError(
                f"BUSI class '{class_name}' not found. Available classes: {self.CLASSES}"
            )
        if index < 0:
            raise ValueError("sample index must be >= 0")

        with self.db.session_scope() as session:
            total_samples = int(
                session.scalar(
                    select(func.count(BusiSampleORM.id)).where(
                        BusiSampleORM.class_name == class_name
                    )
                )
                or 0
            )
            if total_samples <= 0:
                raise FileNotFoundError(
                    f"No BUSI images found for class '{class_name}' in database storage."
                )

            resolved_index = int(index % total_samples)
            sample = session.scalars(
                select(BusiSampleORM)
                .where(BusiSampleORM.class_name == class_name)
                .order_by(BusiSampleORM.image_filename)
                .offset(resolved_index)
                .limit(1)
            ).first()

        if sample is None:
            raise FileNotFoundError(
                f"Could not fetch BUSI sample for class '{class_name}' at index {index}."
            )

        image_rgb = self._decode_rgb_blob(sample.image_blob)
        mask = self._decode_mask_blob(
            sample.mask_blob,
            shape=(int(image_rgb.shape[0]), int(image_rgb.shape[1])),
        )

        return BusiSampleRecord(
            class_name=class_name,
            requested_index=int(index),
            resolved_index=resolved_index,
            total_samples=total_samples,
            image_path=self.config.busi_dir / class_name / sample.image_filename,
            image_rgb=image_rgb,
            mask=mask,
        )

    def list_busi_training_samples(
        self, include_normal: bool = False
    ) -> list[BusiTrainingSampleRecord]:
        classes = self.CLASSES if include_normal else ("benign", "malignant")

        with self.db.session_scope() as session:
            rows = session.scalars(
                select(BusiSampleORM)
                .where(BusiSampleORM.class_name.in_(classes))
                .order_by(BusiSampleORM.class_name, BusiSampleORM.image_filename)
            ).all()

        samples: list[BusiTrainingSampleRecord] = []
        for row in rows:
            class_name = row.class_name
            split = row.split
            if class_name not in self.CLASSES:
                continue
            if split not in {"train", "test"}:
                continue
            samples.append(
                BusiTrainingSampleRecord(
                    sample_id=int(row.id),
                    class_name=cast(Literal["benign", "malignant", "normal"], class_name),
                    label=int(row.label),
                    split=cast(Literal["train", "test"], split),
                    image_rgb=self._decode_rgb_blob(row.image_blob),
                )
            )
        return samples

    def save_busi_training_run(self, run: BusiTrainingRunRecord) -> BusiTrainingRunRecord:
        payload = run.model_dump_json()
        with self.db.session_scope() as session:
            row = BusiTrainingRunORM(
                include_normal=run.include_normal,
                train_accuracy=float(run.train_accuracy),
                test_accuracy=float(run.test_accuracy),
                payload_json=payload,
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Could not persist BUSI training run.")
            run_id = int(row.id)
        return run.model_copy(update={"run_id": run_id})

    def get_latest_busi_training_run(
        self, include_normal: bool = False
    ) -> BusiTrainingRunRecord | None:
        with self.db.session_scope() as session:
            row = session.scalars(
                select(BusiTrainingRunORM)
                .where(BusiTrainingRunORM.include_normal == include_normal)
                .order_by(BusiTrainingRunORM.id.desc())
                .limit(1)
            ).first()

        if row is None:
            return None

        try:
            parsed = BusiTrainingRunRecord.model_validate_json(row.payload_json)
        except Exception:
            return None
        return parsed.model_copy(update={"run_id": int(row.id)})

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
                        {
                            "depth_m": defect.depth_m,
                            "amplitude": defect.amplitude,
                        }
                        for defect in sample.defects
                    ],
                }
            )
        return rows
