#!/usr/bin/env python3
"""Download the Liver Ultrasound Detection dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset

This dataset is publicly downloadable (no Kaggle credentials needed).
It contains 735 annotated liver ultrasound images (Benign/Malignant/Normal)
with polygon segmentation masks that are converted to bounding boxes.

Usage:
  python scripts/download_liver_ultrasound_detection.py
  python scripts/download_liver_ultrasound_detection.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from ultrasound.data.liver_dataset import (
    download_liver_dataset,
    summarize_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Liver Ultrasound Detection dataset")
    parser.add_argument(
        "--dest",
        type=str,
        default=str(_project_root / "data" / "liver_ultrasound_detection"),
        help="Destination directory",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    dest = Path(args.dest)
    print(f"Downloading liver ultrasound dataset to {dest} ...")

    paths = download_liver_dataset(dest, force=args.force)

    summary = summarize_dataset(paths)
    print("\n✓ Dataset ready! Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
