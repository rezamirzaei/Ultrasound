"""Tests for dataset and synthetic data utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ultrasound import data as data_module
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
        class_dir = output_dir / class_name
        image_files = sorted(p for p in class_dir.glob("*.png") if "_mask" not in p.stem)
        assert image_files

        image_path = image_files[0]
        mask_path = class_dir / f"{image_path.stem}_mask.png"
        assert mask_path.exists()

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


def test_dataset_rejects_invalid_split(tmp_path: Path):
    create_sample_data(str(tmp_path / "synthetic"), num_samples=1)

    with pytest.raises(ValueError, match="split must be one of"):
        BUSIDataset(root_dir=str(tmp_path / "synthetic"), split="dev")


def test_dataset_construction_does_not_reset_global_numpy_rng(tmp_path: Path):
    output_dir = tmp_path / "synthetic"
    create_sample_data(str(output_dir), num_samples=2)
    np.random.seed(123)
    expected_second = np.random.RandomState(123).rand(2)[1]
    _ = np.random.rand()

    BUSIDataset(root_dir=str(output_dir), split="train")

    actual_second = np.random.rand()
    assert actual_second == pytest.approx(expected_second)


def test_create_sample_data_requires_positive_sample_count(tmp_path: Path):
    with pytest.raises(ValueError, match="num_samples must be positive"):
        create_sample_data(str(tmp_path / "synthetic"), num_samples=0)


def test_default_transforms_require_positive_image_size() -> None:
    with pytest.raises(ValueError, match="image_size must be positive"):
        BUSIDataset.get_default_transforms(image_size=0)


def test_dataset_split_partitioning_covers_train_val_and_test(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"
    create_sample_data(str(output_dir), num_samples=4)

    train_dataset = BUSIDataset(root_dir=str(output_dir), split="train")
    val_dataset = BUSIDataset(root_dir=str(output_dir), split="val")
    test_dataset = BUSIDataset(root_dir=str(output_dir), split="test")

    assert len(train_dataset) == 5
    assert len(val_dataset) == 1
    assert len(test_dataset) == 2
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == 8


def test_dataset_defaults_to_tensor_loading_and_zero_mask_when_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    class_dir = dataset_root / "benign"
    class_dir.mkdir(parents=True, exist_ok=True)

    for index, fill in enumerate((64, 128)):
        Image.fromarray(np.full((10, 12, 3), fill, dtype=np.uint8), mode="RGB").save(
            class_dir / f"benign_{index:03d}.png"
        )

    dataset = BUSIDataset(root_dir=str(dataset_root), split="train")

    image, mask, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 10, 12)
    assert mask.shape == (1, 10, 12)
    assert int(mask.sum().item()) == 0
    assert label == 0


def test_default_transforms_support_augmentation_and_resize_masks() -> None:
    transform, mask_transform = BUSIDataset.get_default_transforms(image_size=32, augment=True)

    image = transform(Image.fromarray(np.full((24, 18, 3), 120, dtype=np.uint8), mode="RGB"))
    mask = mask_transform(Image.fromarray(np.full((24, 18), 255, dtype=np.uint8), mode="L"))

    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)


def test_download_busi_dataset_reports_existing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "downloads" / "Dataset_BUSI_with_GT"
    dataset_path.mkdir(parents=True, exist_ok=True)

    resolved = data_module.download_busi_dataset(str(tmp_path / "downloads"))

    captured = capsys.readouterr()
    assert resolved == str(dataset_path)
    assert "already exists" in captured.out


def test_download_busi_dataset_prints_manual_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resolved = data_module.download_busi_dataset(str(tmp_path / "downloads"))

    captured = capsys.readouterr()
    assert resolved.endswith("Dataset_BUSI_with_GT")
    assert "BUSI Dataset Download Instructions" in captured.out
    assert "kaggle datasets download" in captured.out


def test_synthetic_generation_helpers_return_expected_shapes() -> None:
    image = data_module._generate_synthetic_ultrasound("benign", size=(32, 48))
    malignant_image, malignant_mask = data_module._generate_synthetic_ultrasound_with_mask(
        "malignant",
        size=(32, 48),
    )
    mask = data_module._generate_synthetic_mask((20, 30))

    assert image.shape == (32, 48, 3)
    assert image.dtype == np.uint8
    assert malignant_image.shape == (32, 48, 3)
    assert malignant_mask.shape == (32, 48)
    assert malignant_mask.dtype == np.uint8
    assert int(mask.max()) == 255
    assert mask.shape == (20, 30)


def test_get_dataloader_validates_inputs_and_uses_runtime_pin_memory(monkeypatch) -> None:
    dataset = torch.utils.data.TensorDataset(torch.randn(3, 2), torch.tensor([0, 1, 0]))
    monkeypatch.setattr(data_module.torch.cuda, "is_available", lambda: False)

    loader = data_module.get_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

    assert loader.batch_size == 2
    assert loader.pin_memory is False

    with pytest.raises(ValueError, match="batch_size must be positive"):
        data_module.get_dataloader(dataset, batch_size=0)
    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        data_module.get_dataloader(dataset, num_workers=-1)
