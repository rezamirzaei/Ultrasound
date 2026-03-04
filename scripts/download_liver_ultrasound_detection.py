#!/usr/bin/env python3
"""Download the Kaggle Liver Ultrasound Detection competition data.

Competition: https://www.kaggle.com/competitions/liver-ultrasound-detection/data

Authentication (one of):
  1. Place credentials in ``~/.kaggle/kaggle.json``
  2. Set env vars ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``

If no credentials are available, use ``--synthetic`` to generate a small
demo dataset for smoke-testing the training pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project ``src/`` is importable when running as a script.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from ultrasound.api.services.liver_dataset import (
    create_synthetic_liver_dataset,
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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a small synthetic dataset (no Kaggle credentials needed)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of synthetic samples to create (only with --synthetic)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)

    if args.synthetic:
        print(f"Creating synthetic liver ultrasound dataset at {dest} ...")
        paths = create_synthetic_liver_dataset(dest, n_samples=args.n_samples)
    else:
        print(f"Downloading liver-ultrasound-detection to {dest} ...")
        paths = download_liver_dataset(dest, force=args.force)

    summary = summarize_dataset(paths)
    print("\nDataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
