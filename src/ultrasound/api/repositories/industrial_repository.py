"""Industrial dataset-specific repository operations."""

from __future__ import annotations

import hashlib
import re
from typing import Literal, cast

from sqlalchemy import delete, func, select

from ultrasound.api.database.models import IndustrialSampleORM, IndustrialTrainingRunORM
from ultrasound.api.models.domain import (
    IndustrialSampleRecord,
    IndustrialTrainingRunRecord,
    IndustrialTrainingSampleRecord,
    IndustrialUploadRecord,
)
from ultrasound.api.repositories.dataset_support import DatasetRepositorySupport


class IndustrialRepository(DatasetRepositorySupport):
    """Persist and read industrial image datasets and training artifacts."""

    def sync_industrial_from_filesystem(self) -> int:
        fingerprint = self._compute_industrial_fingerprint()
        if self._meta_get("industrial_fingerprint") == fingerprint:
            return 0

        inserted = 0
        with self.db.session_scope() as session:
            session.execute(delete(IndustrialSampleORM))
            for dataset_name, split, class_name, image_path, annotation_path in self._collect_industrial_sources():
                image_blob, width, height = self._canonical_png_rgb(image_path)
                annotation_blob = annotation_path.read_bytes() if annotation_path is not None and annotation_path.exists() else None
                try:
                    relative_path = str(image_path.resolve().relative_to(self.config.data_dir.resolve()))
                except ValueError:
                    relative_path = str(image_path.resolve())
                source_hash = hashlib.sha256(image_blob + (annotation_blob or b"") + relative_path.encode("utf-8")).hexdigest()
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
            self._set_meta_value(session, "industrial_fingerprint", fingerprint)
        return inserted

    def _ensure_industrial_seeded(self) -> None:
        fingerprint = self._meta_get("industrial_fingerprint")
        if fingerprint:
            return
        with self.db.session_scope() as session:
            existing = int(session.scalar(select(func.count(IndustrialSampleORM.id))) or 0)
        if existing > 0:
            return
        self.sync_industrial_from_filesystem()

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
                f"Invalid split '{split}' for dataset '{normalized_dataset}'. Expected one of {sorted(allowed_splits)}."
            )
        normalized_class = re.sub(r"[^a-zA-Z0-9_-]+", "_", class_name.strip().lower()).strip("_")
        if not normalized_class:
            raise ValueError("class_name must not be empty")

        image_png, width, height = self._canonical_png_rgb_bytes(image_blob)
        safe_filename = self._safe_filename(
            image_filename,
            default_stem=f"{normalized_dataset}_{normalized_split}_{normalized_class}_uploaded",
        )
        relative_path = f"uploads/{normalized_dataset}/{normalized_split}/{normalized_class}/{safe_filename}"
        source_hash = hashlib.sha256(image_png + (annotation_blob or b"") + relative_path.encode("utf-8")).hexdigest()

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
            dataset_name=cast(Literal["steel_defect", "neu_surface", "casting_defect"], normalized_dataset),
            split=normalized_split,
            class_name=normalized_class,
            image_filename=safe_filename,
            relative_path=relative_path,
            has_annotation=annotation_blob is not None,
            total_class_samples=total_class_samples,
            created_at=created_at,
        )

    def list_industrial_training_samples(
        self, dataset_name: str
    ) -> tuple[list[IndustrialTrainingSampleRecord], dict[str, int], list[str]]:
        self._ensure_industrial_seeded()
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid industrial dataset '{dataset_name}'. Expected one of {self.INDUSTRIAL_DATASETS}."
            )

        with self.db.session_scope() as session:
            rows = session.scalars(
                select(IndustrialSampleORM)
                .where(IndustrialSampleORM.dataset_name == normalized_dataset)
                .order_by(IndustrialSampleORM.split, IndustrialSampleORM.class_name, IndustrialSampleORM.image_filename)
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
                    dataset_name=cast(Literal["steel_defect", "neu_surface", "casting_defect"], normalized_dataset),
                    class_name=class_name,
                    label=int(label),
                    split=cast(Literal["train", "test"], normalized_split),
                    image_rgb=self._decode_rgb_blob(row.image_blob),
                    annotation_blob=row.annotation_blob,
                )
            )

        if len(parsed_samples) < 2:
            raise ValueError(f"Dataset '{normalized_dataset}' needs at least 2 samples for train/test evaluation.")

        n_train = sum(1 for sample in parsed_samples if sample.split == "train")
        n_test = sum(1 for sample in parsed_samples if sample.split == "test")
        if n_train <= 0 or n_test <= 0:
            rebuilt: list[IndustrialTrainingSampleRecord] = []
            train_cutoff = max(1, int(round(0.8 * len(parsed_samples))))
            for index, sample in enumerate(parsed_samples):
                rebuilt_split: Literal["train", "test"] = "train" if index < train_cutoff else "test"
                rebuilt.append(sample.model_copy(update={"split": rebuilt_split}))
            parsed_samples = rebuilt

        return parsed_samples, class_counts, class_names

    def get_industrial_counts(self) -> dict[str, dict[str, dict[str, int]]]:
        self._ensure_industrial_seeded()
        counts: dict[str, dict[str, dict[str, int]]] = {dataset_name: {} for dataset_name in self.INDUSTRIAL_DATASETS}
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
                f"Invalid industrial dataset '{dataset_name}'. Expected one of {self.INDUSTRIAL_DATASETS}."
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
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Available: {self.INDUSTRIAL_DATASETS}")
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
                    f"No industrial samples found for {dataset_name}/{split}/{class_name} in database storage."
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
                f"Could not fetch industrial sample for {dataset_name}/{split}/{class_name} at index {index}."
            )

        return IndustrialSampleRecord(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            requested_index=int(index),
            resolved_index=resolved_index,
            total_samples=total_samples,
            relative_path=sample.relative_path,
            image_rgb=self._decode_rgb_blob(sample.image_blob),
            annotation_blob=sample.annotation_blob,
            has_annotation=sample.annotation_blob is not None,
        )

    def save_industrial_training_run(self, run: IndustrialTrainingRunRecord) -> IndustrialTrainingRunRecord:
        with self.db.session_scope() as session:
            row = IndustrialTrainingRunORM(
                dataset_name=run.dataset_name,
                train_accuracy=float(run.train_accuracy),
                test_accuracy=float(run.test_accuracy),
                payload_json=run.model_dump_json(),
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Could not persist industrial training run.")
            run_id = int(row.id)
        return run.model_copy(update={"run_id": run_id})

    def get_latest_industrial_training_run(self, dataset_name: str) -> IndustrialTrainingRunRecord | None:
        normalized_dataset = dataset_name.strip().lower()
        if normalized_dataset not in self.INDUSTRIAL_DATASETS:
            raise ValueError(
                f"Invalid industrial dataset '{dataset_name}'. Expected one of {self.INDUSTRIAL_DATASETS}."
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
