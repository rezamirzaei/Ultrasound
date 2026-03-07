"""Shared transcranial hydrophone fixtures for phase-retrieval tests."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np


class _CaseParams(TypedDict):
    center_row: int
    center_col: int
    delay: float
    sigma_xy: float
    amplitude: float
    phase_shift: float
    attenuation: float


def _make_case(
    *,
    center_row: int,
    center_col: int,
    delay: float,
    sigma_xy: float,
    amplitude: float,
    phase_shift: float,
    attenuation: float,
) -> np.ndarray:
    n_samples = 384
    n_rows = 32
    n_cols = 32
    time = np.arange(n_samples, dtype=np.float64)
    signal = np.zeros((n_samples, n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        for col in range(n_cols):
            spatial = np.exp(
                -(
                    ((row - center_row) ** 2 + (col - center_col) ** 2)
                    / (2.0 * sigma_xy**2)
                )
            )
            local_delay = delay + 0.12 * (row - center_row) - 0.08 * (col - center_col)
            shifted_time = time - local_delay
            envelope = np.exp(-0.5 * (shifted_time / 14.0) ** 2)
            carrier = np.cos(2.0 * np.pi * 0.10 * shifted_time + phase_shift)
            harmonic = 0.12 * np.cos(2.0 * np.pi * 0.18 * shifted_time + 0.5 * phase_shift)
            pulse = envelope * (carrier + harmonic)
            echo = attenuation * np.exp(-0.5 * ((shifted_time - 34.0) / 18.0) ** 2) * np.cos(
                2.0 * np.pi * 0.08 * (shifted_time - 34.0)
            )
            signal[:, row, col] = amplitude * spatial * (pulse + echo)

    return signal


def create_transcranial_fixture(data_root: Path) -> Path:
    """Create a tiny ETH-like hydrophone scan dataset with deterministic strong traces."""
    root = data_root / "phase_retrieval" / "transcranial" / "Scan_data"
    root.mkdir(parents=True, exist_ok=True)
    cases: dict[str, _CaseParams] = {
        "Parietal_free_field_0_XY": dict(
            center_row=15,
            center_col=20,
            delay=136.0,
            sigma_xy=4.2,
            amplitude=1.00,
            phase_shift=0.25,
            attenuation=0.03,
        ),
        "Frontal_free_field_0_XY": dict(
            center_row=18,
            center_col=19,
            delay=142.0,
            sigma_xy=4.8,
            amplitude=0.92,
            phase_shift=0.45,
            attenuation=0.04,
        ),
        "Frontal_40_XY": dict(
            center_row=12,
            center_col=24,
            delay=148.0,
            sigma_xy=5.2,
            amplitude=0.78,
            phase_shift=0.72,
            attenuation=0.10,
        ),
        "Frontal_0_XY": dict(
            center_row=20,
            center_col=17,
            delay=150.0,
            sigma_xy=5.4,
            amplitude=0.74,
            phase_shift=0.88,
            attenuation=0.12,
        ),
        "Parietal_free_field_0_XZ": dict(
            center_row=14,
            center_col=22,
            delay=138.0,
            sigma_xy=4.6,
            amplitude=0.97,
            phase_shift=0.33,
            attenuation=0.03,
        ),
    }

    for case_name, params in cases.items():
        path = root / f"{case_name}.mat"
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("sigMat", data=_make_case(**params))
    return root.parent
