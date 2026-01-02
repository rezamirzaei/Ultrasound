#!/usr/bin/env python3
"""
Download real ultrasound A-scan / RF data from public sources.

Public Ultrasound Datasets:
1. PICMUS - Plane-wave Imaging Challenge (medical ultrasound RF data)
2. CUBDL - Challenge on Ultrasound Beamforming (raw channel data)
3. OpenBUFF - Open B-mode Ultrasound Freehand Framework
"""

import os
import zipfile
import requests
from pathlib import Path
import urllib.request


def download_file(url, dest_path, description=""):
    """Download a file with progress indicator."""
    print(f"Downloading {description or url}...")
    try:
        # Use requests for better error handling
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        print(f"\n✓ Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def download_picmus_data(data_dir):
    """
    Download PICMUS (Plane-wave Imaging Challenge in Medical UltraSound) data.

    Contains:
    - Simulation phantom RF data
    - Experimental phantom RF data
    - In-vivo carotid RF data
    """
    picmus_dir = data_dir / "picmus"
    picmus_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "simulation_contrast": "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download/simulation_contrast_RF.zip",
        "simulation_resolution": "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download/simulation_resolution_RF.zip",
        "experiments_contrast": "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download/experiments_contrast_RF.zip",
        "experiments_resolution": "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download/experiments_resolution_RF.zip",
    }

    print("\n" + "="*60)
    print("PICMUS Dataset")
    print("="*60)

    for name, url in datasets.items():
        zip_path = picmus_dir / f"{name}.zip"
        extract_dir = picmus_dir / name

        if extract_dir.exists() and len(list(extract_dir.glob("*"))) > 0:
            print(f"✓ {name} already exists")
            continue

        if download_file(url, zip_path, name):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
                zip_path.unlink()
                print(f"  Extracted to {extract_dir}")
            except zipfile.BadZipFile:
                print(f"  Warning: {name} is not a valid zip file")
                # Keep the file anyway, might be raw data
                zip_path.rename(extract_dir / f"{name}.dat")


def download_ius2017_data(data_dir):
    """
    Download IUS 2017 Challenge data (if available).
    """
    ius_dir = data_dir / "ius2017"
    ius_dir.mkdir(parents=True, exist_ok=True)

    # Note: These URLs may require registration
    print("\n" + "="*60)
    print("IUS 2017 Data")
    print("="*60)
    print("Note: Some datasets require registration at:")
    print("  https://www.creatis.insa-lyon.fr/EvasionProject")


def generate_sample_ndt_data(data_dir):
    """
    Generate realistic NDT ultrasound test data.

    This simulates typical industrial NDT scenarios:
    - Weld inspection
    - Thickness measurement
    - Defect detection
    """
    import numpy as np

    ndt_dir = data_dir / "ndt_samples"
    ndt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Generating NDT Sample Data")
    print("="*60)

    # Parameters
    fs = 50e6  # 50 MHz sampling
    fc = 5e6   # 5 MHz transducer
    c = 5900   # Steel: ~5900 m/s

    # Generate different test scenarios
    scenarios = {
        "steel_plate_10mm": {
            "thickness": 0.010,  # 10mm
            "defects": [],
            "description": "Clean 10mm steel plate"
        },
        "steel_plate_with_crack": {
            "thickness": 0.020,  # 20mm
            "defects": [(0.012, 0.3)],  # Crack at 12mm, 30% amplitude
            "description": "20mm steel with internal crack"
        },
        "weld_inspection": {
            "thickness": 0.015,  # 15mm
            "defects": [(0.008, 0.4), (0.011, 0.2)],  # Multiple defects
            "description": "Weld with lack of fusion"
        },
        "corrosion_thinning": {
            "thickness": 0.008,  # Thinned from 10mm to 8mm
            "defects": [],
            "description": "Corroded plate (original 10mm)"
        }
    }

    for name, params in scenarios.items():
        print(f"  Generating: {name}")

        duration = 20e-6  # 20 µs
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # Generate pulse
        pulse_duration = 0.5e-6
        t_pulse = np.arange(0, pulse_duration, 1/fs)
        pulse = np.exp(-((t_pulse - pulse_duration/2)**2) / (2*(pulse_duration/6)**2))
        pulse *= np.sin(2 * np.pi * fc * t_pulse)

        # Initialize signal
        rf = np.zeros(n_samples)

        # Add front wall echo
        fw_idx = int(0.5e-6 * fs)  # Small delay
        if fw_idx + len(pulse) < n_samples:
            rf[fw_idx:fw_idx+len(pulse)] = pulse

        # Add defect echoes
        for defect_depth, defect_amp in params["defects"]:
            defect_time = 2 * defect_depth / c
            defect_idx = int(defect_time * fs)
            if defect_idx + len(pulse) < n_samples:
                rf[defect_idx:defect_idx+len(pulse)] += defect_amp * pulse

        # Add back wall echo
        bw_time = 2 * params["thickness"] / c
        bw_idx = int(bw_time * fs)
        if bw_idx + len(pulse) < n_samples:
            rf[bw_idx:bw_idx+len(pulse)] += 0.8 * pulse

        # Add noise
        rf += 0.05 * np.random.randn(n_samples)

        # Save as numpy file
        np.savez(ndt_dir / f"{name}.npz",
                 rf=rf,
                 time=t,
                 fs=fs,
                 fc=fc,
                 c=c,
                 description=params["description"],
                 thickness=params["thickness"],
                 defects=params["defects"])

    print(f"✓ Generated {len(scenarios)} NDT test files in {ndt_dir}")


def main():
    # Get project data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "ascan_signals"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ULTRASOUND A-SCAN DATA DOWNLOADER")
    print("="*60)
    print(f"Data directory: {data_dir}")

    # Download available datasets
    download_picmus_data(data_dir)
    download_ius2017_data(data_dir)

    # Generate NDT sample data
    generate_sample_ndt_data(data_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Count files
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"  {subdir.name}: {len(files)} files")

    print("\n✓ Data download complete!")
    print("\nNote: Some datasets require manual registration.")
    print("Visit: https://www.creatis.insa-lyon.fr/EvaluationPlatform/")


if __name__ == "__main__":
    main()

