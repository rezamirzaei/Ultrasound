"""Service to ingest uploaded datasets directly into SQL storage."""

from __future__ import annotations

from pathlib import Path

from ultrasound.api.models.domain import BusiUploadRecord, IndustrialUploadRecord
from ultrasound.api.services.interfaces import BusiUploadRepository, IndustrialUploadRepository
from ultrasound.api.services.service_errors import InvalidRequestError


def _normalize_upload_filename(filename: str) -> str:
    normalized = Path((filename or "").replace("\\", "/").strip()).name.strip()
    if not normalized or normalized in {".", ".."}:
        raise InvalidRequestError("Image filename must not be empty")
    return normalized


def _validate_optional_blob(blob: bytes | None, *, label: str) -> None:
    if blob is not None and not blob:
        raise InvalidRequestError(f"{label} file is empty")


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
        _validate_optional_blob(mask_blob, label="Mask")
        normalized_filename = _normalize_upload_filename(image_filename)
        try:
            return self.busi_repository.add_busi_uploaded_sample(
                class_name=class_name,
                split=split,
                image_filename=normalized_filename,
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
        _validate_optional_blob(annotation_blob, label="Annotation")
        normalized_filename = _normalize_upload_filename(image_filename)
        try:
            return self.industrial_repository.add_industrial_uploaded_sample(
                dataset_name=dataset_name,
                split=split,
                class_name=class_name,
                image_filename=normalized_filename,
                image_blob=image_blob,
                annotation_blob=annotation_blob,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
