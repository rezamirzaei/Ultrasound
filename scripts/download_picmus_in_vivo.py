#!/usr/bin/env python3
"""Download the PICMUS in-vivo raw ultrasound dataset.

Dataset: https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home

Usage:
  python scripts/download_picmus_in_vivo.py
  python scripts/download_picmus_in_vivo.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from ultrasound.data.picmus_dataset import (  # noqa: E402
    PICMUS_SOURCE_URL,
    download_picmus_in_vivo,
    list_picmus_rf_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PICMUS in-vivo ultrasound RF data")
    parser.add_argument(
        "--dest",
        type=str,
        default=str(_project_root / "data" / "picmus"),
        help="Destination directory",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    root = download_picmus_in_vivo(args.dest, force=args.force)
    cases = list_picmus_rf_cases(root)

    print(f"PICMUS source: {PICMUS_SOURCE_URL}")
    print(f"Dataset root: {root}")
    print(f"Cases: {', '.join(cases) if cases else 'none'}")


if __name__ == "__main__":
    main()
