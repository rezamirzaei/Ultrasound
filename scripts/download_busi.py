#!/usr/bin/env python3
"""
BUSI Dataset Download Script
Downloads the Breast Ultrasound Images Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
"""

import shutil
from pathlib import Path


def download_busi_dataset(data_dir: str = "data/busi"):
    """Download and extract the BUSI dataset.

    This function supports two setups:

    1) Kaggle API (recommended)
       - Requires a valid Kaggle token at ``~/.kaggle/kaggle.json``.

    2) Manual ZIP placement
       - Place the downloaded Kaggle ZIP under ``data/busi/`` and re-run this script.

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

    # Manual zip fallback
    zip_candidates = sorted(data_path.glob("*.zip"))
    if zip_candidates:
        import zipfile

        zip_path = zip_candidates[0]
        print(f"Found a ZIP file at {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)

        dataset_subdir = data_path / "Dataset_BUSI_with_GT"
        if dataset_subdir.exists():
            for folder in ["benign", "malignant", "normal"]:
                src = dataset_subdir / folder
                dst = data_path / folder
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
            shutil.rmtree(dataset_subdir)

        print(f"✓ Extracted BUSI dataset to {data_path}")
        return data_path

    print("Downloading BUSI dataset from Kaggle...")
    print("=" * 50)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            "aryashah2k/breast-ultrasound-images-dataset",
            path=str(data_path),
            unzip=True,
        )

        dataset_subdir = data_path / "Dataset_BUSI_with_GT"
        if dataset_subdir.exists():
            for folder in ["benign", "malignant", "normal"]:
                src = dataset_subdir / folder
                dst = data_path / folder
                if src.exists():
                    if dst.exists():
                        continue
                    shutil.move(str(src), str(dst))
            shutil.rmtree(dataset_subdir)

        print(f"✓ Dataset downloaded to {data_path}")

    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("\nManual download option:")
        print("- Download the dataset ZIP from Kaggle and place it under:")
        print(f"  {data_path}")
        print("- Then re-run this script to extract and arrange folders.")
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

    out = download_busi_dataset(str(data_dir))
    ok = verify_dataset(str(data_dir))

    if not ok:
        raise SystemExit(
            "BUSI verification failed. Ensure data/busi contains benign/, malignant/, normal/."
        )
