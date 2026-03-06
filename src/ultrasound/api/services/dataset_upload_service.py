"""Service to ingest uploaded datasets directly into SQL storage."""

from __future__ import annotations

from ultrasound.api.models.domain import BusiUploadRecord, IndustrialUploadRecord
from ultrasound.api.services.interfaces import BusiUploadRepository, IndustrialUploadRepository
from ultrasound.api.services.service_errors import InvalidRequestError


class DatasetUploadService:
    """Coordinates validated upload ingestion into ORM-backed dataset tables."""

    def __init__(
        self,
        busi_repository: BusiUploadRepository,
        industrial_repository: IndustrialUploadRepository,
    ) -> None:
        self.busi_repository = busi_repository
        self.industrial_repository = industrial_repository

    def upload_busi_sample(
        self,
        class_name: str,
        split: str,
        image_filename: str,
        image_blob: bytes,
        mask_blob: bytes | None,
    ) -> BusiUploadRecord:
        if not image_blob:
            raise InvalidRequestError("Image file is empty")
        try:
            return self.busi_repository.add_busi_uploaded_sample(
                class_name=class_name,
                split=split,
                image_filename=image_filename,
                image_blob=image_blob,
                mask_blob=mask_blob,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

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
            raise InvalidRequestError("Image file is empty")
        try:
            return self.industrial_repository.add_industrial_uploaded_sample(
                dataset_name=dataset_name,
                split=split,
                class_name=class_name,
                image_filename=image_filename,
                image_blob=image_blob,
                annotation_blob=annotation_blob,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
