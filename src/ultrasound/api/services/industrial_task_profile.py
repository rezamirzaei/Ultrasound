"""Task-profile resolution for industrial dataset responses."""

from __future__ import annotations

from typing import Literal

IndustrialTaskType = Literal[
    "classification_single_label",
    "classification_single_label_with_bbox",
]
IndustrialClassificationMode = Literal["binary", "multiclass"]
IndustrialLabelSource = Literal["folder_name", "folder_name_plus_xml_bbox"]


def resolve_industrial_task_profile(
    dataset_name: str,
    class_labels: list[str],
    annotated_samples: int,
) -> tuple[
    IndustrialTaskType,
    IndustrialClassificationMode,
    IndustrialLabelSource,
    bool,
    str,
]:
    segmentation_supported = int(annotated_samples) > 0
    task_type: IndustrialTaskType
    label_source: IndustrialLabelSource
    if segmentation_supported:
        task_type = "classification_single_label_with_bbox"
        label_source = "folder_name_plus_xml_bbox"
    else:
        task_type = "classification_single_label"
        label_source = "folder_name"

    classification_mode: IndustrialClassificationMode = (
        "binary" if len(class_labels) == 2 else "multiclass"
    )

    if segmentation_supported:
        segmentation_notes = (
            "Annotation labels are available: segmentation masks are derived from XML "
            "bounding boxes."
        )
    elif dataset_name == "casting_defect":
        segmentation_notes = (
            "Casting dataset is classification-only in current storage "
            "(def_front vs ok_front); no segmentation labels found."
        )
    elif dataset_name == "steel_defect":
        segmentation_notes = (
            "Steel dataset labels come from train/valid/test class folders; "
            "Thumbs.db is ignored and does not provide supervised labels."
        )
    else:
        segmentation_notes = (
            "No annotation labels were found in current SQL storage; "
            "segmentation is unavailable for this dataset."
        )

    return (
        task_type,
        classification_mode,
        label_source,
        segmentation_supported,
        segmentation_notes,
    )
