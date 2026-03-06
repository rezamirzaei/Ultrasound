"""
Dataset utilities for downloading and loading ultrasound imaging datasets.

Supported Datasets:
- BUSI: Breast Ultrasound Images Dataset (Kaggle)
- CAMUS: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class _BusiSample(TypedDict):
    image_path: str
    mask_paths: list[str]
    class_name: str
    label: int


class BUSIDataset(Dataset):
    """
    Breast Ultrasound Images Dataset (BUSI)

    Dataset for breast ultrasound image classification and segmentation.
    Contains benign, malignant, and normal cases with corresponding masks.

    Reference:
        Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020).
        Dataset of breast ultrasound images. Data in Brief, 28, 104863.

    Download from: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
    """

    CLASSES = ["benign", "malignant", "normal"]
    CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}
    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: transforms.Compose | None = None,
        mask_transform: transforms.Compose | None = None,
        include_normal: bool = False,
        binary_classification: bool = True,
    ):
        """
        Initialize BUSI dataset.

        Args:
            root_dir: Path to dataset root directory
            split: 'train', 'val', or 'test'
            transform: Transforms for input images
            mask_transform: Transforms for segmentation masks
            include_normal: Whether to include normal (non-lesion) cases
            binary_classification: If True, treat as benign vs malignant classification
        """
        self.root_dir = Path(root_dir)
        self.split = self._normalize_split(split)
        self.transform = transform
        self.mask_transform = mask_transform
        self.include_normal = include_normal
        self.binary_classification = binary_classification

        self.samples = self._load_samples()

    @classmethod
    def _normalize_split(cls, split: str) -> str:
        normalized = split.strip().lower()
        if normalized not in cls.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(cls.VALID_SPLITS)}")
        return normalized

    def _load_samples(self) -> list[_BusiSample]:
        """Load and organize dataset samples."""
        samples: list[_BusiSample] = []

        classes = self.CLASSES if self.include_normal else ["benign", "malignant"]

        for class_name in classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # Get all image files (excluding masks)
            image_files = sorted([f for f in class_dir.glob("*.png") if "_mask" not in f.stem])

            for img_path in image_files:
                # Find corresponding mask(s)
                mask_pattern = f"{img_path.stem}_mask*.png"
                mask_files = sorted(class_dir.glob(mask_pattern))

                label = self.CLASS_TO_IDX[class_name]
                if self.binary_classification and not self.include_normal:
                    label = 0 if class_name == "benign" else 1

                samples.append(
                    {
                        "image_path": str(img_path),
                        "mask_paths": [str(m) for m in mask_files],
                        "class_name": class_name,
                        "label": label,
                    }
                )

        # Split dataset
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(samples))

        train_end = int(0.7 * len(samples))
        val_end = int(0.85 * len(samples))

        if self.split == "train":
            indices = indices[:train_end]
        elif self.split == "val":
            indices = indices[train_end:val_end]
        else:  # test
            indices = indices[val_end:]

        return [samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        # Load image
        with Image.open(str(sample["image_path"])) as pil_image:
            image = pil_image.convert("RGB")

        # Load and combine masks
        if sample["mask_paths"]:
            masks: list[np.ndarray] = []
            for mask_path in sample["mask_paths"]:
                with Image.open(str(mask_path)) as pil_mask:
                    masks.append(np.array(pil_mask.convert("L")))
            mask = np.maximum.reduce(masks) if len(masks) > 1 else masks[0]
            mask = Image.fromarray(mask)
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Keep masks binary after geometric transforms.
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()

        return image, mask, int(sample["label"])

    @staticmethod
    def get_default_transforms(image_size: int = 256, augment: bool = False):
        """Get default image transforms."""
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if augment:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )

        return transform, mask_transform


def download_busi_dataset(download_dir: str) -> str:
    """
    Instructions for downloading the BUSI dataset.

    The dataset needs to be downloaded from Kaggle:
    https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

    Steps:
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)
    3. Run: kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
    4. Extract to the specified directory

    Args:
        download_dir: Directory to save the dataset

    Returns:
        Path to the extracted dataset
    """
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    dataset_path = download_path / "Dataset_BUSI_with_GT"

    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return str(dataset_path)

    print("=" * 60)
    print("BUSI Dataset Download Instructions")
    print("=" * 60)
    print(f"""
To download the Breast Ultrasound Images Dataset:

Option 1: Using Kaggle CLI
--------------------------
1. Install Kaggle: pip install kaggle
2. Set up credentials: Create ~/.kaggle/kaggle.json with your API key
3. Download: kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
4. Extract to: {download_dir}

Option 2: Manual Download
-------------------------
1. Visit: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
2. Download the dataset ZIP file
3. Extract to: {download_dir}

The dataset contains:
- 780 ultrasound images (PNG format)
- 3 classes: benign (437), malignant (210), normal (133)
- Ground truth segmentation masks
""")

    return str(dataset_path)


def create_sample_data(output_dir: str, num_samples: int = 10) -> str:
    """
    Create synthetic ultrasound-like sample data for testing.

    This generates synthetic images that mimic ultrasound characteristics
    for testing the pipeline when real data is not available.

    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of samples per class

    Returns:
        Path to generated data
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    output_path = Path(output_dir)
    rng = np.random.default_rng(42)

    for class_name in ["benign", "malignant"]:
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            # Generate synthetic ultrasound-like image
            img, mask = _generate_synthetic_ultrasound_with_mask(class_name, rng=rng)

            # Save image and mask
            Image.fromarray(img).save(class_dir / f"{class_name}_{i:03d}.png")
            Image.fromarray(mask).save(class_dir / f"{class_name}_{i:03d}_mask.png")

    print(f"Created synthetic dataset at {output_path}")
    return str(output_path)


def _generate_synthetic_ultrasound(
    class_name: str, size: tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Generate a synthetic ultrasound-like image with speckle noise."""
    image, _ = _generate_synthetic_ultrasound_with_mask(class_name, size=size)
    return image


def _generate_synthetic_ultrasound_with_mask(
    class_name: str,
    size: tuple[int, int] = (256, 256),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic ultrasound-like image and its aligned lesion mask."""
    if rng is None:
        rng = np.random.default_rng()

    # Base grayscale image
    img = rng.normal(100, 30, size).astype(np.float32)

    # Add depth-dependent attenuation (darker at bottom)
    depth_gradient = np.linspace(1.0, 0.6, size[0])[:, np.newaxis]
    img = img * depth_gradient

    # Add speckle noise (multiplicative)
    speckle = rng.exponential(1.0, size)
    img = img * speckle

    # Add a lesion-like structure
    center_y = size[0] // 2 + int(rng.integers(-30, 31))
    center_x = size[1] // 2 + int(rng.integers(-30, 31))

    if class_name == "malignant":
        # Malignant: irregular, hypoechoic (darker)
        radius = int(rng.integers(20, 40))
        y, x = np.ogrid[: size[0], : size[1]]
        # Add some irregularity
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        angle = np.arctan2(y - center_y, x - center_x)
        irregular_radius = radius * (1 + 0.3 * np.sin(5 * angle))
        lesion_mask = dist < irregular_radius
        img[lesion_mask] = img[lesion_mask] * 0.4
    else:
        # Benign: more regular, oval shape
        radius_y = int(rng.integers(15, 30))
        radius_x = int(rng.integers(20, 40))
        y, x = np.ogrid[: size[0], : size[1]]
        lesion_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 < 1
        img[lesion_mask] = img[lesion_mask] * 0.5

    # Normalize to 0-255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert to RGB (grayscale in all channels)
    img_rgb = np.stack([img, img, img], axis=-1)

    mask = lesion_mask.astype(np.uint8) * 255
    return img_rgb, mask


def _generate_synthetic_mask(size: tuple[int, int]) -> np.ndarray:
    """Generate a simple circular mask."""
    mask = np.zeros(size, dtype=np.uint8)
    center_y, center_x = size[0] // 2, size[1] // 2
    radius = min(size) // 6

    y, x = np.ogrid[: size[0], : size[1]]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask[dist < radius] = 255

    return mask


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
