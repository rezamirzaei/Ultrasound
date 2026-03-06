"""BUSI-specific repository operations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import delete, func, select

from ultrasound.api.database.models import BusiSampleORM, BusiTrainingRunORM
from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
    BusiUploadRecord,
)
from ultrasound.api.repositories.dataset_support import DatasetRepositorySupport


class BusiRepository(DatasetRepositorySupport):
    """Persist and read BUSI samples, uploads, and training runs."""

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
                image_paths = sorted(path for path in class_dir.glob("*.png") if "_mask" not in path.stem)
                train_cutoff = self._resolve_train_cutoff(len(image_paths))
                for index, image_path in enumerate(image_paths):
                    image_blob, width, height = self._canonical_png_rgb(image_path)
                    mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                    mask_blob = self._canonical_png_mask(
                        mask_candidates[0] if mask_candidates else None,
                        width=width,
                        height=height,
                    )
                    source_hash = hashlib.sha256(image_blob + (mask_blob or b"" )).hexdigest()
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
            self._set_meta_value(session, "busi_fingerprint", fingerprint)
        return inserted

    def get_busi_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.CLASSES}
        with self.db.session_scope() as session:
            rows = session.execute(
                select(BusiSampleORM.class_name, func.count(BusiSampleORM.id)).group_by(BusiSampleORM.class_name)
            ).all()
        for class_name, total in rows:
            if class_name in counts:
                counts[class_name] = int(total)
        return counts

    def get_busi_sample(self, class_name: str, index: int = 0) -> BusiSampleRecord:
        if class_name not in self.CLASSES:
            raise FileNotFoundError(f"BUSI class '{class_name}' not found. Available classes: {self.CLASSES}")
        if index < 0:
            raise ValueError("sample index must be >= 0")

        with self.db.session_scope() as session:
            total_samples = int(
                session.scalar(select(func.count(BusiSampleORM.id)).where(BusiSampleORM.class_name == class_name)) or 0
            )
            if total_samples <= 0:
                raise FileNotFoundError(f"No BUSI images found for class '{class_name}' in database storage.")
            resolved_index = int(index % total_samples)
            sample = session.scalars(
                select(BusiSampleORM)
                .where(BusiSampleORM.class_name == class_name)
                .order_by(BusiSampleORM.image_filename)
                .offset(resolved_index)
                .limit(1)
            ).first()

        if sample is None:
            raise FileNotFoundError(f"Could not fetch BUSI sample for class '{class_name}' at index {index}.")

        image_rgb = self._decode_rgb_blob(sample.image_blob)
        mask = self._decode_mask_blob(sample.mask_blob, shape=(int(image_rgb.shape[0]), int(image_rgb.shape[1])))
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
        safe_filename = self._safe_filename(image_filename, default_stem=f"{normalized_class}_uploaded")
        source_hash = hashlib.sha256(image_png + (mask_png or b"" )).hexdigest()

        with self.db.session_scope() as session:
            row = session.scalars(
                select(BusiSampleORM)
                .where(BusiSampleORM.class_name == normalized_class, BusiSampleORM.image_filename == safe_filename)
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
                session.scalar(select(func.count(BusiSampleORM.id)).where(BusiSampleORM.class_name == normalized_class)) or 0
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

    def list_busi_training_samples(self, include_normal: bool = False) -> list[BusiTrainingSampleRecord]:
        classes = self.CLASSES if include_normal else ("benign", "malignant")
        with self.db.session_scope() as session:
            rows = session.scalars(
                select(BusiSampleORM)
                .where(BusiSampleORM.class_name.in_(classes))
                .order_by(BusiSampleORM.class_name, BusiSampleORM.image_filename)
            ).all()

        samples: list[BusiTrainingSampleRecord] = []
        for row in rows:
            if row.class_name not in self.CLASSES or row.split not in {"train", "test"}:
                continue
            samples.append(
                BusiTrainingSampleRecord(
                    sample_id=int(row.id),
                    class_name=cast(Literal["benign", "malignant", "normal"], row.class_name),
                    label=int(row.label),
                    split=cast(Literal["train", "test"], row.split),
                    image_rgb=self._decode_rgb_blob(row.image_blob),
                )
            )
        return samples

    def save_busi_training_run(self, run: BusiTrainingRunRecord) -> BusiTrainingRunRecord:
        with self.db.session_scope() as session:
            row = BusiTrainingRunORM(
                include_normal=run.include_normal,
                train_accuracy=float(run.train_accuracy),
                test_accuracy=float(run.test_accuracy),
                payload_json=run.model_dump_json(),
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Could not persist BUSI training run.")
            run_id = int(row.id)
        return run.model_copy(update={"run_id": run_id})

    def get_latest_busi_training_run(self, include_normal: bool = False) -> BusiTrainingRunRecord | None:
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
