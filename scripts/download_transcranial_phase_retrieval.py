#!/usr/bin/env python3
"""Download the ETH transcranial hydrophone scan dataset used for phase retrieval.

Dataset:
https://www.research-collection.ethz.ch/entities/dataset/77f541ab-2cc2-4274-a9f6-49c5475754ca

Usage:
  python scripts/download_transcranial_phase_retrieval.py
  python scripts/download_transcranial_phase_retrieval.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from ultrasound.data.transcranial_phase_dataset import (  # noqa: E402
    ETH_TRANSCRANIAL_SOURCE_URL,
    download_transcranial_dataset,
    list_transcranial_scan_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ETH transcranial hydrophone scan data for phase retrieval"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(_project_root / "data" / "phase_retrieval"),
        help="Destination directory",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    root = download_transcranial_dataset(args.dest, force=args.force)
    cases = list_transcranial_scan_cases(root)

    print(f"Source: {ETH_TRANSCRANIAL_SOURCE_URL}")
    print(f"Dataset root: {root}")
    print(f"Cases: {', '.join(cases) if cases else 'none'}")


if __name__ == "__main__":
    main()
