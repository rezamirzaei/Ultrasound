"""Reusable NDT A-scan analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, hilbert


@dataclass(frozen=True)
class NdtAscanAnalysisResult:
    envelope_db: np.ndarray
    peak_indices: np.ndarray
    peak_times_us: list[float]
    freq_hz: np.ndarray
    spectrum_db: np.ndarray
    report: dict[str, Any]
    status: dict[str, Any]


def run_ndt_ascan_analysis(
    rf: np.ndarray,
    time_s: np.ndarray,
    *,
    fs_hz: float,
    fc_hz: float,
    c_mps: float,
    nominal_thickness_m: float | None,
    seed: int = 42,
    peak_threshold_ratio: float = 0.2,
    min_peak_distance_us: float = 0.4,
) -> NdtAscanAnalysisResult:
    rf_arr = np.asarray(rf, dtype=np.float64).reshape(-1)
    time_arr = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if rf_arr.size == 0 or time_arr.size != rf_arr.size:
        raise ValueError("rf and time_s must be non-empty arrays with the same length")

    envelope = np.abs(hilbert(rf_arr))
    envelope_db = 20.0 * np.log10(envelope / (np.max(envelope) + 1e-12) + 1e-12)
    peak_threshold = float(peak_threshold_ratio) * float(np.max(envelope))
    min_dist = max(10, int(float(min_peak_distance_us) * 1e-6 * float(fs_hz)))
    peak_indices, _ = find_peaks(envelope, height=peak_threshold, distance=min_dist)

    estimated_thickness_mm = float("nan")
    if peak_indices.size >= 2:
        tof = float(time_arr[int(peak_indices[1])] - time_arr[int(peak_indices[0])])
        estimated_thickness_mm = 0.5 * float(c_mps) * tof * 1e3

    nominal_thickness_mm = (
        float(nominal_thickness_m * 1e3) if nominal_thickness_m is not None else float("nan")
    )
    thickness_error_mm = float(estimated_thickness_mm - nominal_thickness_mm)

    n = int(rf_arr.size)
    window = np.hanning(n)
    spectrum = np.fft.rfft(rf_arr * window)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    spectrum_db = 20.0 * np.log10(
        np.abs(spectrum) / (np.max(np.abs(spectrum)) + 1e-12) + 1e-12
    )

    peak_times_us = [float(v) for v in (time_arr[peak_indices] * 1e6).round(6).tolist()]
    status = {
        "echoes_detected": int(peak_indices.size),
        "thickness_estimate_available": bool(np.isfinite(estimated_thickness_mm)),
        "abs_thickness_error_mm": (
            float(abs(thickness_error_mm)) if np.isfinite(thickness_error_mm) else float("nan")
        ),
    }
    status["overall_pass"] = bool(
        int(status["echoes_detected"]) >= 2 and status["thickness_estimate_available"]
    )

    report = {
        "seed": int(seed),
        "fs_hz": float(fs_hz),
        "fc_hz": float(fc_hz),
        "c_mps": float(c_mps),
        "n_points": n,
        "detected_echoes": int(peak_indices.size),
        "peak_times_us": peak_times_us,
        "nominal_thickness_mm": nominal_thickness_mm,
        "estimated_thickness_mm": float(estimated_thickness_mm),
        "thickness_error_mm": thickness_error_mm,
    }
    return NdtAscanAnalysisResult(
        envelope_db=envelope_db,
        peak_indices=peak_indices,
        peak_times_us=peak_times_us,
        freq_hz=freq_hz,
        spectrum_db=spectrum_db,
        report=report,
        status=status,
    )
