#!/usr/bin/env python3
"""Verify all downloaded datasets."""
from pathlib import Path

data_path = Path(__file__).parent.parent / "data"

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

datasets = {
    "BUSI (Medical Ultrasound)": {
        "path": data_path / "busi",
        "extensions": ["*.png"],
        "exclude": "_mask"
    },
    "Steel Defect (NEU)": {
        "path": data_path / "steel_defect",
        "extensions": ["*.jpg", "*.png", "*.bmp"],
        "exclude": None
    },
    "Casting Defect": {
        "path": data_path / "casting_defect",
        "extensions": ["*.jpeg", "*.png"],
        "exclude": None
    },
    "NEU Surface Defect": {
        "path": data_path / "neu_surface",
        "extensions": ["*.jpg", "*.png", "*.bmp"],
        "exclude": None
    },
}

total = 0
for name, config in datasets.items():
    path = config["path"]
    if path.exists():
        count = 0
        for ext in config["extensions"]:
            files = list(path.rglob(ext))
            if config["exclude"]:
                files = [f for f in files if config["exclude"] not in f.stem]
            count += len(files)
        total += count
        status = f"✓ {count:,} images"
    else:
        status = "✗ Not found"
    print(f"{name:30} {status}")

print("-" * 60)
print(f"{'TOTAL':30} {total:,} images")
print("=" * 60)

