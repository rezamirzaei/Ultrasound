"""Tests for upload validation and repository error mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from ultrasound.api.models.domain import BusiUploadRecord, IndustrialUploadRecord
from ultrasound.api.services.dataset_upload_service import DatasetUploadService
from ultrasound.api.services.service_errors import InvalidRequestError


class _BusiUploadRepositoryStub:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.error: Exception | None = None

    def add_busi_uploaded_sample(self, **kwargs: Any) -> BusiUploadRecord:
        if self.error is not None:
            raise self.error
        self.last_kwargs = kwargs
        return BusiUploadRecord(
            sample_id=1,
            class_name="benign",
            split="train",
            image_filename=str(kwargs["image_filename"]),
            total_class_samples=4,
            created_at=datetime.now(tz=timezone.utc),
        )


class _IndustrialUploadRepositoryStub:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.error: Exception | None = None

    def add_industrial_uploaded_sample(self, **kwargs: Any) -> IndustrialUploadRecord:
        if self.error is not None:
            raise self.error
        self.last_kwargs = kwargs
        return IndustrialUploadRecord(
            sample_id=2,
            dataset_name="steel_defect",
            split="train",
            class_name="crazing",
            image_filename=str(kwargs["image_filename"]),
            relative_path="uploads/steel_defect/train/crazing/sample.png",
            has_annotation=bool(kwargs.get("annotation_blob")),
            total_class_samples=2,
            created_at=datetime.now(tz=timezone.utc),
        )


def test_busi_upload_normalizes_filename_before_persisting() -> None:
    busi_repository = _BusiUploadRepositoryStub()
    service = DatasetUploadService(cast(Any, busi_repository), cast(Any, _IndustrialUploadRepositoryStub()))

    record = service.upload_busi_sample(
        class_name="benign",
        split="train",
        image_filename=r" C:\fakepath\scan image.png ",
        image_blob=b"image",
        mask_blob=b"mask",
    )

    assert record.image_filename == "scan image.png"
    assert busi_repository.last_kwargs is not None
    assert busi_repository.last_kwargs["image_filename"] == "scan image.png"


@pytest.mark.parametrize(
    ("image_filename", "image_blob", "mask_blob", "message"),
    [
        ("sample.png", b"", None, "Image file is empty"),
        ("   ", b"image", None, "Image filename must not be empty"),
        ("sample.png", b"image", b"", "Mask file is empty"),
    ],
)
def test_busi_upload_rejects_invalid_payloads(
    image_filename: str,
    image_blob: bytes,
    mask_blob: bytes | None,
    message: str,
) -> None:
    service = DatasetUploadService(
        cast(Any, _BusiUploadRepositoryStub()),
        cast(Any, _IndustrialUploadRepositoryStub()),
    )

    with pytest.raises(InvalidRequestError, match=message):
        service.upload_busi_sample("benign", "train", image_filename, image_blob, mask_blob)


def test_industrial_upload_normalizes_filename_and_annotation_blob() -> None:
    industrial_repository = _IndustrialUploadRepositoryStub()
    service = DatasetUploadService(cast(Any, _BusiUploadRepositoryStub()), cast(Any, industrial_repository))

    record = service.upload_industrial_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="/tmp/uploads/specimen.jpg",
        image_blob=b"image",
        annotation_blob=b"<xml/>",
    )

    assert record.image_filename == "specimen.jpg"
    assert industrial_repository.last_kwargs is not None
    assert industrial_repository.last_kwargs["image_filename"] == "specimen.jpg"


@pytest.mark.parametrize(
    ("image_filename", "image_blob", "annotation_blob", "message"),
    [
        ("sample.png", b"", None, "Image file is empty"),
        ("", b"image", None, "Image filename must not be empty"),
        ("sample.png", b"image", b"", "Annotation file is empty"),
    ],
)
def test_industrial_upload_rejects_invalid_payloads(
    image_filename: str,
    image_blob: bytes,
    annotation_blob: bytes | None,
    message: str,
) -> None:
    service = DatasetUploadService(
        cast(Any, _BusiUploadRepositoryStub()),
        cast(Any, _IndustrialUploadRepositoryStub()),
    )

    with pytest.raises(InvalidRequestError, match=message):
        service.upload_industrial_sample(
            "steel_defect",
            "train",
            "crazing",
            image_filename,
            image_blob,
            annotation_blob,
        )


def test_upload_service_maps_repository_value_errors() -> None:
    busi_repository = _BusiUploadRepositoryStub()
    industrial_repository = _IndustrialUploadRepositoryStub()
    busi_repository.error = ValueError("invalid busi sample")
    industrial_repository.error = ValueError("invalid industrial sample")
    service = DatasetUploadService(cast(Any, busi_repository), cast(Any, industrial_repository))

    with pytest.raises(InvalidRequestError, match="invalid busi sample"):
        service.upload_busi_sample("benign", "train", "sample.png", b"image", None)

    with pytest.raises(InvalidRequestError, match="invalid industrial sample"):
        service.upload_industrial_sample(
            "steel_defect",
            "train",
            "crazing",
            "sample.png",
            b"image",
            None,
        )
