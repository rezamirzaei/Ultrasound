"""Shared PICMUS-like HDF5 fixtures for tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def create_picmus_fixture(data_root: Path) -> Path:
    """Create a tiny PICMUS-like RF dataset with deterministic high-energy windows."""
    root = data_root / "picmus" / "in_vivo"
    cases = {
        "carotid_long": (1, 2, 72, 1.0),
        "carotid_cross": (0, 1, 108, 1.2),
    }

    for case_name, (angle_index, element_index, start_index, amplitude) in cases.items():
        case_dir = root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / f"{case_name}_expe_dataset_rf.hdf5"
        with h5py.File(path, "w") as h5_file:
            dataset = h5_file.create_group("US").create_group("US_DATASET0000")
            data = dataset.create_group("data")
            real = np.zeros((2, 3, 256), dtype=np.float32)
            burst = amplitude * np.sin(np.linspace(0.0, 8.0 * np.pi, 96, dtype=np.float32))
            real[angle_index, element_index, start_index : start_index + burst.size] = burst
            real[0, 0, 8:104] = 0.15 * np.cos(np.linspace(0.0, 6.0 * np.pi, 96, dtype=np.float32))
            data.create_dataset("real", data=real)
            data.create_dataset("imag", data=np.zeros_like(real))
            dataset.create_dataset("sampling_frequency", data=np.array([20e6], dtype=np.float32))
            dataset.create_dataset("sound_speed", data=np.array([1540.0], dtype=np.float32))
            dataset.create_dataset("probe_geometry", data=np.zeros((3, 3), dtype=np.float32))
    return root
