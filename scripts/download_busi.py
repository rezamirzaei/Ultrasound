#!/usr/bin/env python3
"""
BUSI Dataset Download Script
Downloads the Breast Ultrasound Images Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def download_busi_dataset(data_dir: str = "data/busi"):
    """
    Download and extract the BUSI dataset.

    The dataset contains 780 images:
    - Normal: 133 images
    - Benign: 437 images
    - Malignant: 210 images

    Each image has a corresponding segmentation mask.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (data_path / "benign").exists() and (data_path / "malignant").exists():
        print(f"✓ BUSI dataset already exists at {data_path}")
        return data_path

    print("Downloading BUSI dataset from Kaggle...")
    print("=" * 50)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.dataset_download_files(
            'aryashah2k/breast-ultrasound-images-dataset',
            path=str(data_path),
            unzip=True
        )

        # Reorganize files if needed
        dataset_subdir = data_path / "Dataset_BUSI_with_GT"
        if dataset_subdir.exists():
            for folder in ["benign", "malignant", "normal"]:
                src = dataset_subdir / folder
                dst = data_path / folder
                if src.exists():
                    shutil.move(str(src), str(dst))
            shutil.rmtree(dataset_subdir)

        print(f"✓ Dataset downloaded to {data_path}")

    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nTo download manually:")
        print("1. Go to: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print("2. Download and extract to: data/busi/")
        print("3. Ensure folders exist: data/busi/benign/, data/busi/malignant/, data/busi/normal/")
        return None

    return data_path


def verify_dataset(data_dir: str = "data/busi"):
    """Verify the dataset structure and count images."""
    data_path = Path(data_dir)

    print("\nDataset Verification:")
    print("=" * 50)

    total = 0
    for category in ["benign", "malignant", "normal"]:
        folder = data_path / category
        if folder.exists():
            images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            # Filter out mask images
            images = [f for f in images if "_mask" not in f.stem]
            count = len(images)
            total += count
            print(f"  {category.capitalize():12} {count:4} images")
        else:
            print(f"  {category.capitalize():12} NOT FOUND")

    print(f"  {'Total':12} {total:4} images")
    return total > 0


if __name__ == "__main__":
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    data_dir = project_root / "data" / "busi"

    download_busi_dataset(str(data_dir))
    verify_dataset(str(data_dir))

