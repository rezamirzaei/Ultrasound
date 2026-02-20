"""Tests for dataset and synthetic data utilities."""

from pathlib import Path

import numpy as np
from PIL import Image

from ultrasound.data import BUSIDataset, create_sample_data


def test_create_sample_data_outputs_images_and_masks(tmp_path: Path):
    output_dir = tmp_path / "synthetic"
    create_sample_data(str(output_dir), num_samples=2)

    for class_name in ("benign", "malignant"):
        class_dir = output_dir / class_name
        images = sorted(p for p in class_dir.glob("*.png") if "_mask" not in p.stem)
        masks = sorted(class_dir.glob("*_mask.png"))
        assert len(images) == 2
        assert len(masks) == 2


def test_synthetic_mask_matches_darker_lesion_region(tmp_path: Path):
    output_dir = tmp_path / "synthetic"
    create_sample_data(str(output_dir), num_samples=1)

    for class_name in ("benign", "malignant"):
        image_path = next(
            p for p in (output_dir / class_name).glob("*.png") if "_mask" not in p.stem
        )
        mask_path = Path(str(image_path).replace(".png", "_mask.png"))

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
        mask = np.array(Image.open(mask_path), dtype=np.uint8) > 0

        inside_mean = float(image[mask].mean())
        outside_mean = float(image[~mask].mean())
        assert inside_mean < outside_mean


def test_dataset_masks_are_binary_after_transforms(tmp_path: Path):
    output_dir = tmp_path / "synthetic"
    create_sample_data(str(output_dir), num_samples=2)
    transform, mask_transform = BUSIDataset.get_default_transforms(image_size=64, augment=False)

    dataset = BUSIDataset(
        root_dir=str(output_dir),
        split="train",
        transform=transform,
        mask_transform=mask_transform,
    )
    _, mask, _ = dataset[0]
    unique_vals = set(mask.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})
