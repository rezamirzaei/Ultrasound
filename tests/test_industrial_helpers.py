"""Tests for focused industrial training helper modules."""

from __future__ import annotations

import numpy as np

from ultrasound.api.models.domain import IndustrialTrainingSampleRecord
from ultrasound.api.services.industrial_feature_extractor import IndustrialFeatureExtractor
from ultrasound.api.services.industrial_segmentation_baseline import IndustrialSegmentationBaseline
from ultrasound.api.services.industrial_task_profile import resolve_industrial_task_profile


def _sample_record(
    sample_id: int,
    image_rgb: np.ndarray,
    annotation_blob: bytes | None = None,
) -> IndustrialTrainingSampleRecord:
    return IndustrialTrainingSampleRecord(
        sample_id=sample_id,
        dataset_name="neu_surface",
        class_name="crazing",
        label=0,
        split="train",
        image_rgb=image_rgb,
        annotation_blob=annotation_blob,
    )


def test_industrial_feature_extractor_returns_stable_feature_vector() -> None:
    image = np.zeros((12, 10, 3), dtype=np.uint8)
    image[2:10, 3:8] = [32, 128, 255]

    features = IndustrialFeatureExtractor().extract(image)

    assert features.shape == (4617,)
    assert features.dtype == np.float32


def test_resolve_industrial_task_profile_prefers_annotation_aware_mode() -> None:
    (
        task_type,
        classification_mode,
        label_source,
        segmentation_supported,
        segmentation_notes,
    ) = resolve_industrial_task_profile(
        dataset_name="neu_surface",
        class_labels=["crazing", "patches"],
        annotated_samples=3,
    )

    assert task_type == "classification_single_label_with_bbox"
    assert classification_mode == "binary"
    assert label_source == "folder_name_plus_xml_bbox"
    assert segmentation_supported is True
    assert "Annotation labels are available" in segmentation_notes


def test_resolve_industrial_task_profile_reports_classification_only_steel_notes() -> None:
    (
        task_type,
        classification_mode,
        label_source,
        segmentation_supported,
        segmentation_notes,
    ) = resolve_industrial_task_profile(
        dataset_name="steel_defect",
        class_labels=["crazing", "inclusion", "patches"],
        annotated_samples=0,
    )

    assert task_type == "classification_single_label"
    assert classification_mode == "multiclass"
    assert label_source == "folder_name"
    assert segmentation_supported is False
    assert "Thumbs.db" in segmentation_notes


def test_segmentation_baseline_parses_bbox_xml() -> None:
    baseline = IndustrialSegmentationBaseline()
    xml = (
        b"<annotation><object><bndbox><xmin>2.4</xmin><ymin>1.6</ymin>"
        b"<xmax>8.2</xmax><ymax>7.9</ymax></bndbox></object></annotation>"
    )

    mask, bbox_count = baseline.mask_from_annotation(xml, shape=(10, 10))

    assert bbox_count == 1
    assert int(mask.sum()) == 36 * 255


def test_segmentation_baseline_handles_malformed_xml() -> None:
    baseline = IndustrialSegmentationBaseline()

    mask, bbox_count = baseline.mask_from_annotation(b"<annotation>", shape=(8, 8))

    assert bbox_count == 0
    np.testing.assert_array_equal(mask, np.zeros((8, 8), dtype=np.uint8))


def test_segmentation_baseline_scores_annotated_samples() -> None:
    baseline = IndustrialSegmentationBaseline()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:10, 5:11] = 255
    xml = (
        b"<annotation><object><bndbox><xmin>5</xmin><ymin>4</ymin>"
        b"<xmax>11</xmax><ymax>10</ymax></bndbox></object></annotation>"
    )
    samples = [
        _sample_record(1, image_rgb=image, annotation_blob=xml),
        _sample_record(2, image_rgb=image, annotation_blob=None),
    ]

    iou, dice, annotated_samples = baseline.baseline_metrics(samples)

    assert annotated_samples == 1
    assert iou is not None and 0.0 <= iou <= 1.0
    assert dice is not None and 0.0 <= dice <= 1.0
