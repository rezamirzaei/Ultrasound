"""Service to ingest uploaded datasets directly into SQL storage."""

from __future__ import annotations

from ultrasound.api.models.domain import BusiUploadRecord, IndustrialUploadRecord
from ultrasound.api.services.interfaces import DatasetUploadRepository


class DatasetUploadService:
    """Coordinates validated upload ingestion into ORM-backed dataset tables."""

    def __init__(self, dataset_repository: DatasetUploadRepository):
        self.dataset_repository = dataset_repository

    def upload_busi_sample(
        self,
        class_name: str,
        split: str,
        image_filename: str,
        image_blob: bytes,
        mask_blob: bytes | None,
    ) -> BusiUploadRecord:
        if not image_blob:
            raise ValueError("Image file is empty")
        return self.dataset_repository.add_busi_uploaded_sample(
            class_name=class_name,
            split=split,
            image_filename=image_filename,
            image_blob=image_blob,
            mask_blob=mask_blob,
        )

    def upload_industrial_sample(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        image_filename: str,
        image_blob: bytes,
        annotation_blob: bytes | None,
    ) -> IndustrialUploadRecord:
        if not image_blob:
            raise ValueError("Image file is empty")
        return self.dataset_repository.add_industrial_uploaded_sample(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            image_filename=image_filename,
            image_blob=image_blob,
            annotation_blob=annotation_blob,
        )
