"""Repository layer for SQLAlchemy-backed dataset access."""

from __future__ import annotations

import hashlib
import re
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
    IndustrialSampleORM,
    IndustrialTrainingRunORM,
    NdtDefectORM,
    NdtSampleORM,
)
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
    BusiUploadRecord,
    IndustrialSampleRecord,
    IndustrialTrainingRunRecord,
    IndustrialTrainingSampleRecord,
    IndustrialUploadRecord,
    NdtDefectRecord,
    NdtSampleRecord,
)


class DatasetRepository:
    """Encapsulates dataset access and metadata extraction."""

    CLASSES = ("benign", "malignant", "normal")
    CLASS_TO_LABEL = {"benign": 0, "malignant": 1, "normal": 2}
    INDUSTRIAL_DATASETS = ("steel_defect", "neu_surface", "casting_defect")
    INDUSTRIAL_SPLITS = {
        "steel_defect": {"train", "valid", "test"},
        "neu_surface": {"train", "validation"},
        "casting_defect": {"train", "test", "full"},
    }

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

    def _canonical_png_rgb_bytes(self, image_blob: bytes) -> tuple[bytes, int, int]:
        with Image.open(BytesIO(image_blob)) as pil_image:
            image_rgb = pil_image.convert("RGB")
            width, height = image_rgb.size
            buffer = BytesIO()
            image_rgb.save(buffer, format="PNG")
        return buffer.getvalue(), int(width), int(height)

    def _canonical_png_mask_bytes(
        self, mask_blob: bytes | None, width: int, height: int
    ) -> bytes | None:
        if mask_blob is None:
            return None
        with Image.open(BytesIO(mask_blob)) as pil_mask:
            mask_gray = pil_mask.convert("L")
            if mask_gray.size != (width, height):
                mask_gray = mask_gray.resize((width, height), Image.Resampling.NEAREST)
            buffer = BytesIO()
            mask_gray.save(buffer, format="PNG")
        return buffer.getvalue()

    def _safe_filename(self, filename: str, default_stem: str) -> str:
        stem = Path(filename).stem if filename else ""
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
        if not stem:
            stem = default_stem
        return f"{stem}.png"

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

    def _industrial_roots(self) -> dict[str, Path]:
        return {
            "steel_defect": self.config.data_dir / "steel_defect",
            "neu_surface": self.config.data_dir / "neu_surface",
            "casting_defect": self.config.data_dir / "casting_defect",
        }

    def _collect_industrial_sources(self) -> list[tuple[str, str, str, Path, Path | None]]:
        records: list[tuple[str, str, str, Path, Path | None]] = []
        roots = self._industrial_roots()

        steel_root = roots["steel_defect"] / "NEU Metal Surface Defects Data"
        if steel_root.exists():
            for split in ("train", "valid", "test"):
                split_dir = steel_root / split
                if not split_dir.exists():
                    continue
                for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*.bmp")):
                        records.append(
                            (
                                "steel_defect",
                                split,
                                class_dir.name.lower(),
                                image_path,
                                None,
                            )
                        )

        neu_root = roots["neu_surface"] / "NEU-DET"
        if neu_root.exists():
            for split in ("train", "validation"):
                images_root = neu_root / split / "images"
                ann_root = neu_root / split / "annotations"
                if not images_root.exists():
                    continue
                for class_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*")):
                        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                            continue
                        annotation_path = ann_root / f"{image_path.stem}.xml"
                        records.append(
                            (
                                "neu_surface",
                                split,
                                class_dir.name.lower(),
                                image_path,
                                annotation_path if annotation_path.exists() else None,
                            )
                        )

        casting_train_test_root = (
            self.config.data_dir / "casting_defect" / "casting_data" / "casting_data"
        )
        if casting_train_test_root.exists():
            for split in ("train", "test"):
                split_dir = casting_train_test_root / split
                if not split_dir.exists():
                    continue
                for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*.jpeg")):
                        records.append(
                            (
                                "casting_defect",
                                split,
                                class_dir.name.lower(),
                                image_path,
                                None,
                            )
                        )

        casting_flat_root = (
            self.config.data_dir / "casting_defect" / "casting_512x512" / "casting_512x512"
        )
        if casting_flat_root.exists():
            for class_dir in sorted(path for path in casting_flat_root.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.glob("*.jpeg")):
                    records.append(
                        (
                            "casting_defect",
                            "full",
                            class_dir.name.lower(),
                            image_path,
                            None,
                        )
                    )

        return records

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
                digest.update(f"{class_name}:missing|".encode())
                continue

            image_paths = sorted(
                path for path in class_dir.glob("*.png") if "_mask" not in path.stem
            )
            digest.update(f"{class_name}:{len(image_paths)}|".encode())
            for image_path in image_paths:
                stat = image_path.stat()
                digest.update(
                    f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode()
                )
                mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                if mask_candidates:
                    mask_stat = mask_candidates[0].stat()
                    digest.update(
                        f"{mask_candidates[0].name}:{mask_stat.st_size}:{mask_stat.st_mtime_ns}|".encode()
                    )
        return digest.hexdigest()

    def _compute_ndt_fingerprint(self) -> str:
        digest = hashlib.sha256()
        if not self.config.ndt_dir.exists():
            digest.update(b"ndt:missing")
            return digest.hexdigest()

        sample_paths = sorted(self.config.ndt_dir.glob("*.npz"))
        digest.update(f"ndt:{len(sample_paths)}|".encode())
        for sample_path in sample_paths:
            stat = sample_path.stat()
            digest.update(f"{sample_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
        return digest.hexdigest()

    def _compute_industrial_fingerprint(self) -> str:
        digest = hashlib.sha256()
        sources = self._collect_industrial_sources()
        digest.update(f"industrial:{len(sources)}|".encode())
        for dataset_name, split, class_name, image_path, annotation_path in sources:
            image_stat = image_path.stat()
            digest.update(
                (
                    f"{dataset_name}:{split}:{class_name}:{image_path.name}:"
                    f"{image_stat.st_size}:{image_stat.st_mtime_ns}|"
                ).encode()
            )
            if annotation_path is not None:
                ann_stat = annotation_path.stat()
                digest.update(
                    (f"{annotation_path.name}:{ann_stat.st_size}:{ann_stat.st_mtime_ns}|").encode()
                )
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

    def sync_industrial_from_filesystem(self) -> int:
        fingerprint = self._compute_industrial_fingerprint()
        if self._meta_get("industrial_fingerprint") == fingerprint:
            return 0

        inserted = 0
        with self.db.session_scope() as session:
            session.execute(delete(IndustrialSampleORM))
            for (
                dataset_name,
                split,
                class_name,
                image_path,
                annotation_path,
            ) in self._collect_industrial_sources():
                image_blob, width, height = self._canonical_png_rgb(image_path)
                annotation_blob: bytes | None = None
                if annotation_path is not None and annotation_path.exists():
                    annotation_blob = annotation_path.read_bytes()
                try:
                    relative_path = str(
                        image_path.resolve().relative_to(self.config.data_dir.resolve())
                    )
                except ValueError:
                    relative_path = str(image_path.resolve())
                source_hash = hashlib.sha256(
                    image_blob + (annotation_blob or b"") + relative_path.encode("utf-8")
                ).hexdigest()
                session.add(
                    IndustrialSampleORM(
                        dataset_name=dataset_name,
                        split=split,
                        class_name=class_name,
                        image_filename=image_path.name,
                        relative_path=relative_path,
                        image_blob=image_blob,
                        annotation_blob=annotation_blob,
                        width=width,
                        height=height,
                        source_hash=source_hash,
                    )
                )
                inserted += 1

            row = session.get(DatasetMetaORM, "industrial_fingerprint")
            if row is None:
                session.add(DatasetMetaORM(key="industrial_fingerprint", value=fingerprint))
            else:
                row.value = fingerprint

        return inserted

    def _ensure_industrial_seeded(self) -> None:
        """Seed industrial tables once if DB has never been populated."""
        fingerprint = self._meta_get("industrial_fingerprint")
        if fingerprint:
            return
        with self.db.session_scope() as session:
            existing = int(session.scalar(select(func.count(IndustrialSampleORM.id))) or 0)
        if existing > 0:
            return
        self.sync_industrial_from_filesystem()

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

    def add_busi_uploaded_sample(
        self,
        class_name: str,
        split: str,
        image_filename: str,
        image_blob: bytes,
        mask_blob: bytes | None = None,
    ) -> BusiUploadRecord:
        normalized_class = class_name.strip().lower()
        if normalized_class not in self.CLASSES:
            raise ValueError(f"Invalid BUSI class '{class_name}'. Expected one of {self.CLASSES}.")

        normalized_split = split.strip().lower()
        if normalized_split not in {"train", "test"}:
            raise ValueError("BUSI split must be 'train' or 'test'.")

        image_png, width, height = self._canonical_png_rgb_bytes(image_blob)
        mask_png = self._canonical_png_mask_bytes(mask_blob, width=width, height=height)
        safe_filename = self._safe_filename(
            image_filename, default_stem=f"{normalized_class}_uploaded"
        )
        source_hash = hashlib.sha256(image_png + (mask_png or b"")).hexdigest()

        with self.db.session_scope() as session:
            row = session.scalars(
                select(BusiSampleORM)
                .where(
                    BusiSampleORM.class_name == normalized_class,
                    BusiSampleORM.image_filename == safe_filename,
                )
                .limit(1)
            ).first()
            if row is None:
                row = BusiSampleORM(
                    class_name=normalized_class,
                    image_filename=safe_filename,
                    sample_stem=Path(safe_filename).stem,
                    image_blob=image_png,
                    mask_blob=mask_png,
                    width=width,
                    height=height,
                    label=self.CLASS_TO_LABEL[normalized_class],
                    split=normalized_split,
                    source_hash=source_hash,
                )
                session.add(row)
            else:
                row.sample_stem = Path(safe_filename).stem
                row.image_blob = image_png
                row.mask_blob = mask_png
                row.width = width
                row.height = height
                row.label = self.CLASS_TO_LABEL[normalized_class]
                row.split = normalized_split
                row.source_hash = source_hash
            session.flush()

            if row.id is None or row.created_at is None:
                raise RuntimeError("Could not persist uploaded BUSI sample.")

            total_class_samples = int(
                session.scalar(
                    select(func.count(BusiSampleORM.id)).where(
                        BusiSampleORM.class_name == normalized_class
                    )
                )
                or 0
            )
            sample_id = int(row.id)
            created_at = row.created_at

        return BusiUploadRecord(
            sample_id=sample_id,
            class_name=cast(Literal["benign", "malignant", "normal"], normalized_class),
            split=cast(Literal["train", "test"], normalized_split),
            image_filename=safe_filename,
            total_class_samples=total_class_samples,
            created_at=created_at,
        )

    def add_industrial_uploaded_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        image_filename: str,
        image_blob: bytes,
        annotation_blob: bytes | None = None,
    ) -> IndustrialUploadRecord:
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid dataset '{dataset_name}'. Expected one of {self.INDUSTRIAL_DATASETS}."
            )

        normalized_split = split.strip().lower()
        allowed_splits = self.INDUSTRIAL_SPLITS[normalized_dataset]
        if normalized_split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{split}' for dataset '{normalized_dataset}'. "
                f"Expected one of {sorted(allowed_splits)}."
            )

        normalized_class = re.sub(r"[^a-zA-Z0-9_-]+", "_", class_name.strip().lower()).strip("_")
        if not normalized_class:
            raise ValueError("class_name must not be empty")

        image_png, width, height = self._canonical_png_rgb_bytes(image_blob)
        safe_filename = self._safe_filename(
            image_filename,
            default_stem=f"{normalized_dataset}_{normalized_split}_{normalized_class}_uploaded",
        )
        relative_path = (
            f"uploads/{normalized_dataset}/{normalized_split}/{normalized_class}/{safe_filename}"
        )
        source_hash = hashlib.sha256(
            image_png + (annotation_blob or b"") + relative_path.encode("utf-8")
        ).hexdigest()

        with self.db.session_scope() as session:
            row = session.scalars(
                select(IndustrialSampleORM)
                .where(
                    IndustrialSampleORM.dataset_name == normalized_dataset,
                    IndustrialSampleORM.relative_path == relative_path,
                )
                .limit(1)
            ).first()
            if row is None:
                row = IndustrialSampleORM(
                    dataset_name=normalized_dataset,
                    split=normalized_split,
                    class_name=normalized_class,
                    image_filename=safe_filename,
                    relative_path=relative_path,
                    image_blob=image_png,
                    annotation_blob=annotation_blob,
                    width=width,
                    height=height,
                    source_hash=source_hash,
                )
                session.add(row)
            else:
                row.split = normalized_split
                row.class_name = normalized_class
                row.image_filename = safe_filename
                row.image_blob = image_png
                row.annotation_blob = annotation_blob
                row.width = width
                row.height = height
                row.source_hash = source_hash
            session.flush()

            if row.id is None or row.created_at is None:
                raise RuntimeError("Could not persist uploaded industrial sample.")

            total_class_samples = int(
                session.scalar(
                    select(func.count(IndustrialSampleORM.id)).where(
                        IndustrialSampleORM.dataset_name == normalized_dataset,
                        IndustrialSampleORM.split == normalized_split,
                        IndustrialSampleORM.class_name == normalized_class,
                    )
                )
                or 0
            )
            sample_id = int(row.id)
            created_at = row.created_at

        return IndustrialUploadRecord(
            sample_id=sample_id,
            dataset_name=cast(
                Literal["steel_defect", "neu_surface", "casting_defect"],
                normalized_dataset,
            ),
            split=normalized_split,
            class_name=normalized_class,
            image_filename=safe_filename,
            relative_path=relative_path,
            has_annotation=annotation_blob is not None,
            total_class_samples=total_class_samples,
            created_at=created_at,
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

    def list_industrial_training_samples(
        self, dataset_name: str
    ) -> tuple[list[IndustrialTrainingSampleRecord], dict[str, int], list[str]]:
        self._ensure_industrial_seeded()
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid industrial dataset '{dataset_name}'. "
                f"Expected one of {self.INDUSTRIAL_DATASETS}."
            )

        with self.db.session_scope() as session:
            rows = session.scalars(
                select(IndustrialSampleORM)
                .where(IndustrialSampleORM.dataset_name == normalized_dataset)
                .order_by(
                    IndustrialSampleORM.split,
                    IndustrialSampleORM.class_name,
                    IndustrialSampleORM.image_filename,
                )
            ).all()

        if not rows:
            raise ValueError(f"No samples found in SQL storage for dataset '{normalized_dataset}'.")

        class_names = sorted({str(row.class_name) for row in rows})
        class_to_label = {name: idx for idx, name in enumerate(class_names)}

        split_alias = {"train": "train", "test": "test", "validation": "test", "valid": "test"}
        parsed_samples: list[IndustrialTrainingSampleRecord] = []
        class_counts: dict[str, int] = {class_name: 0 for class_name in class_names}

        for row in rows:
            if row.id is None:
                continue
            class_name = str(row.class_name)
            label = class_to_label[class_name]
            normalized_split = split_alias.get(str(row.split).strip().lower(), "train")
            class_counts[class_name] = int(class_counts.get(class_name, 0) + 1)
            parsed_samples.append(
                IndustrialTrainingSampleRecord(
                    sample_id=int(row.id),
                    dataset_name=cast(
                        Literal["steel_defect", "neu_surface", "casting_defect"],
                        normalized_dataset,
                    ),
                    class_name=class_name,
                    label=int(label),
                    split=cast(Literal["train", "test"], normalized_split),
                    image_rgb=self._decode_rgb_blob(row.image_blob),
                    annotation_blob=row.annotation_blob,
                )
            )

        if len(parsed_samples) < 2:
            raise ValueError(
                f"Dataset '{normalized_dataset}' needs at least 2 samples for train/test evaluation."
            )

        n_train = sum(1 for sample in parsed_samples if sample.split == "train")
        n_test = sum(1 for sample in parsed_samples if sample.split == "test")
        if n_train <= 0 or n_test <= 0:
            rebuilt: list[IndustrialTrainingSampleRecord] = []
            for index, sample in enumerate(parsed_samples):
                rebuilt_split: Literal["train", "test"] = (
                    "train" if index < max(1, int(round(0.8 * len(parsed_samples)))) else "test"
                )
                rebuilt.append(sample.model_copy(update={"split": rebuilt_split}))
            parsed_samples = rebuilt

        return parsed_samples, class_counts, class_names

    def get_industrial_counts(self) -> dict[str, dict[str, dict[str, int]]]:
        self._ensure_industrial_seeded()
        counts: dict[str, dict[str, dict[str, int]]] = {
            dataset_name: {} for dataset_name in self.INDUSTRIAL_DATASETS
        }
        with self.db.session_scope() as session:
            rows = session.execute(
                select(
                    IndustrialSampleORM.dataset_name,
                    IndustrialSampleORM.split,
                    IndustrialSampleORM.class_name,
                    func.count(IndustrialSampleORM.id),
                ).group_by(
                    IndustrialSampleORM.dataset_name,
                    IndustrialSampleORM.split,
                    IndustrialSampleORM.class_name,
                )
            ).all()
            for dataset_name, split, class_name, sample_count in rows:
                dataset_bucket = counts.setdefault(str(dataset_name), {})
                split_bucket = dataset_bucket.setdefault(str(split), {})
                split_bucket[str(class_name)] = int(sample_count)
        return counts

    def get_industrial_annotation_count(self, dataset_name: str) -> int:
        self._ensure_industrial_seeded()
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid industrial dataset '{dataset_name}'. "
                f"Expected one of {self.INDUSTRIAL_DATASETS}."
            )

        with self.db.session_scope() as session:
            count = int(
                session.scalar(
                    select(func.count(IndustrialSampleORM.id)).where(
                        IndustrialSampleORM.dataset_name == normalized_dataset,
                        IndustrialSampleORM.annotation_blob.is_not(None),
                    )
                )
                or 0
            )
        return count

    def get_industrial_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        index: int = 0,
    ) -> IndustrialSampleRecord:
        self._ensure_industrial_seeded()
        if dataset_name not in self.INDUSTRIAL_DATASETS:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Available: {self.INDUSTRIAL_DATASETS}"
            )
        if index < 0:
            raise ValueError("sample index must be >= 0")

        with self.db.session_scope() as session:
            total_samples = int(
                session.scalar(
                    select(func.count(IndustrialSampleORM.id)).where(
                        IndustrialSampleORM.dataset_name == dataset_name,
                        IndustrialSampleORM.split == split,
                        IndustrialSampleORM.class_name == class_name,
                    )
                )
                or 0
            )
            if total_samples <= 0:
                raise FileNotFoundError(
                    "No industrial samples found for "
                    f"{dataset_name}/{split}/{class_name} in database storage."
                )

            resolved_index = int(index % total_samples)
            sample = session.scalars(
                select(IndustrialSampleORM)
                .where(
                    IndustrialSampleORM.dataset_name == dataset_name,
                    IndustrialSampleORM.split == split,
                    IndustrialSampleORM.class_name == class_name,
                )
                .order_by(IndustrialSampleORM.image_filename)
                .offset(resolved_index)
                .limit(1)
            ).first()

        if sample is None:
            raise FileNotFoundError(
                "Could not fetch industrial sample for "
                f"{dataset_name}/{split}/{class_name} at index {index}."
            )

        image_rgb = self._decode_rgb_blob(sample.image_blob)
        return IndustrialSampleRecord(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            requested_index=int(index),
            resolved_index=resolved_index,
            total_samples=total_samples,
            relative_path=sample.relative_path,
            image_rgb=image_rgb,
            annotation_blob=sample.annotation_blob,
            has_annotation=sample.annotation_blob is not None,
        )

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

    def save_industrial_training_run(
        self, run: IndustrialTrainingRunRecord
    ) -> IndustrialTrainingRunRecord:
        payload = run.model_dump_json()
        with self.db.session_scope() as session:
            row = IndustrialTrainingRunORM(
                dataset_name=run.dataset_name,
                train_accuracy=float(run.train_accuracy),
                test_accuracy=float(run.test_accuracy),
                payload_json=payload,
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Could not persist industrial training run.")
            run_id = int(row.id)
        return run.model_copy(update={"run_id": run_id})

    def get_latest_industrial_training_run(
        self, dataset_name: str
    ) -> IndustrialTrainingRunRecord | None:
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid industrial dataset '{dataset_name}'. "
                f"Expected one of {self.INDUSTRIAL_DATASETS}."
            )

        with self.db.session_scope() as session:
            row = session.scalars(
                select(IndustrialTrainingRunORM)
                .where(IndustrialTrainingRunORM.dataset_name == normalized_dataset)
                .order_by(IndustrialTrainingRunORM.id.desc())
                .limit(1)
            ).first()

        if row is None:
            return None

        try:
            parsed = IndustrialTrainingRunRecord.model_validate_json(row.payload_json)
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
