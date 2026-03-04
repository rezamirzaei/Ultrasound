#!/usr/bin/env python3
"""Set up Kaggle credentials and download the Liver Ultrasound Detection dataset.

Usage:
  # Option A — pass credentials as args:
  python scripts/setup_kaggle_and_download.py --username YOUR_USER --key YOUR_KEY

  # Option B — export env vars first:
  export KAGGLE_USERNAME=YOUR_USER
  export KAGGLE_KEY=YOUR_KEY
  python scripts/setup_kaggle_and_download.py

  # Option C — already have ~/.kaggle/kaggle.json:
  python scripts/setup_kaggle_and_download.py

To get your Kaggle API key:
  1. Go to https://www.kaggle.com/settings
  2. Scroll to "API" → "Create New Token" (downloads kaggle.json)

You must also accept the competition rules at:
  https://www.kaggle.com/competitions/liver-ultrasound-detection/rules
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent


def ensure_kaggle_credentials(username: str | None = None, key: str | None = None) -> bool:
    """Make sure Kaggle credentials are available. Return True on success."""
    # 1. Env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✓ Kaggle credentials found in environment variables.")
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    # 2. Existing file
    if kaggle_json.exists():
        print(f"✓ Kaggle credentials found at {kaggle_json}")
        return True

    # 3. Provided via CLI args → create the file
    if username and key:
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(
            json.dumps({"username": username, "key": key}, indent=2),
            encoding="utf-8",
        )
        kaggle_json.chmod(stat.S_IRUSR | stat.S_IWUSR)
        print(f"✓ Credentials saved to {kaggle_json}")
        return True

    # 4. Nothing available
    print("✗ Kaggle credentials not found.")
    print()
    print("Provide them in one of these ways:")
    print("  a) --username YOUR_USER --key YOUR_KEY")
    print("  b) export KAGGLE_USERNAME / KAGGLE_KEY")
    print("  c) Place kaggle.json at ~/.kaggle/kaggle.json")
    print()
    print("Get your API token at: https://www.kaggle.com/settings")
    return False


def download_competition_data(dest_dir: Path, *, force: bool = False) -> None:
    """Download and extract the liver-ultrasound-detection competition data."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading liver-ultrasound-detection → {dest_dir} ...")
    api.competition_download_files(
        "liver-ultrasound-detection",
        path=str(dest_dir),
        force=force,
        quiet=False,
    )

    for zp in sorted(dest_dir.glob("*.zip")):
        print(f"Extracting {zp.name} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dest_dir)
        print(f"  ✓ Extracted {zp.name}")


def summarize(dest_dir: Path) -> None:
    """Print dataset summary."""
    print("\nDataset contents:")
    for item in sorted(dest_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            file_count = sum(1 for f in item.rglob("*") if f.is_file())
            print(f"  📁 {item.name}/  ({file_count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Liver Ultrasound Detection dataset")
    parser.add_argument("--username", type=str, help="Kaggle username")
    parser.add_argument("--key", type=str, help="Kaggle API key")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument(
        "--dest",
        type=str,
        default=str(_project_root / "data" / "liver_ultrasound_detection"),
    )
    args = parser.parse_args()
    dest = Path(args.dest)

    # Check if real data already exists (more than 100 images = real data)
    train_dir = dest / "train"
    if train_dir.is_dir() and not args.force:
        train_count = sum(1 for _ in train_dir.iterdir() if _.is_file())
        if train_count > 50:
            print(f"✓ Dataset already exists at {dest} ({train_count} train images)")
            summarize(dest)
            return

    if not ensure_kaggle_credentials(args.username, args.key):
        sys.exit(1)

    try:
        download_competition_data(dest, force=args.force)
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}")
        print("\nMake sure you have:")
        print("  1. Valid Kaggle credentials")
        print("  2. Accepted the competition rules at:")
        print("     https://www.kaggle.com/competitions/liver-ultrasound-detection/rules")
        sys.exit(1)

    summarize(dest)
    print("\n✓ Done! Dataset ready at:", dest)


if __name__ == "__main__":
    main()

