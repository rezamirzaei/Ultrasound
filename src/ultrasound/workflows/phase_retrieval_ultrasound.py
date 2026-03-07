"""Reusable phase-retrieval workflows for real ultrasound waveforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import istft, stft

from ultrasound.data.transcranial_phase_dataset import (
    default_transcranial_case,
    list_transcranial_scan_cases,
    load_transcranial_waveform_window,
    select_high_energy_hydrophone_window,
)


@dataclass(frozen=True)
class PhaseRetrievalResult:
    true_signal: np.ndarray
    recovered_signal: np.ndarray
    measured_spectrogram: np.ndarray
    reconstructed_spectrogram: np.ndarray
    true_phase_spectrum: np.ndarray
    recovered_phase_spectrum: np.ndarray
    residual_curve: list[float]
    report: dict[str, Any]
    status: dict[str, bool]
    signal_metadata: dict[str, Any] | None = None
    scan_energy_map: np.ndarray | None = None


@dataclass(frozen=True)
class PhaseRetrievalTuningResult:
    best_config: dict[str, Any]
    ranked_results: list[dict[str, Any]]
    cases: list[str]


def _stft_complex(signal: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    noverlap = n_fft - hop_length
    _, _, spectrum = stft(
        signal,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros",
    )
    return np.asarray(spectrum, dtype=np.complex128)


def _griffin_lim_from_magnitude(
    magnitude: np.ndarray,
    *,
    signal_length: int,
    n_fft: int,
    hop_length: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    noverlap = n_fft - hop_length
    rng = np.random.default_rng(seed)
    phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=magnitude.shape))
    estimate = magnitude.astype(np.complex128) * phase
    residual_curve: list[float] = []
    _, initial_signal = istft(
        estimate,
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    initial_signal = np.asarray(initial_signal[:signal_length], dtype=np.float64)

    for _ in range(int(n_iter)):
        _, signal_estimate = istft(
            estimate,
            nperseg=n_fft,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )
        spectrum_estimate = _stft_complex(signal_estimate, n_fft=n_fft, hop_length=hop_length)
        rows = min(spectrum_estimate.shape[0], magnitude.shape[0])
        cols = min(spectrum_estimate.shape[1], magnitude.shape[1])
        residual = np.linalg.norm(
            np.abs(spectrum_estimate[:rows, :cols]) - magnitude[:rows, :cols]
        ) / (np.linalg.norm(magnitude[:rows, :cols]) + 1e-12)
        residual_curve.append(float(residual))
        estimate = magnitude.astype(np.complex128)
        estimate[:rows, :cols] *= np.exp(1j * np.angle(spectrum_estimate[:rows, :cols]))

    _, recovered_signal = istft(
        estimate,
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    recovered_signal = np.asarray(recovered_signal[:signal_length], dtype=np.float64)
    reconstructed_spectrogram = _stft_complex(recovered_signal, n_fft=n_fft, hop_length=hop_length)
    return initial_signal, recovered_signal, reconstructed_spectrogram, residual_curve


def _best_aligned_signal(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    *,
    max_shift: int = 32,
) -> tuple[np.ndarray, float, float, int, float]:
    best_error = float("inf")
    best_corr = 0.0
    best_shift = 0
    best_scale = 1.0

    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift >= 0:
            true_window = x_true[shift:]
            recovered_window = x_hat[: len(true_window)]
        else:
            recovered_window = x_hat[-shift:]
            true_window = x_true[: len(recovered_window)]
        if true_window.size < 32:
            continue
        scale = float(
            np.dot(true_window, recovered_window)
            / (np.dot(recovered_window, recovered_window) + 1e-12)
        )
        error = float(
            np.linalg.norm(true_window - scale * recovered_window)
            / (np.linalg.norm(true_window) + 1e-12)
        )
        if error < best_error:
            corr = float(
                abs(
                    np.dot(true_window, recovered_window)
                    / (
                        np.linalg.norm(true_window) * np.linalg.norm(recovered_window) + 1e-12
                    )
                )
            )
            best_error = error
            best_corr = corr
            best_shift = shift
            best_scale = scale

    aligned = np.zeros_like(x_true)
    if best_shift >= 0:
        n = min(x_true.size - best_shift, x_hat.size)
        aligned[best_shift : best_shift + n] = best_scale * x_hat[:n]
    else:
        n = min(x_true.size, x_hat.size + best_shift)
        aligned[:n] = best_scale * x_hat[-best_shift : -best_shift + n]
    return aligned, best_error, best_corr, best_shift, best_scale


def _phase_spectrum(signal: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(np.fft.rfft(signal)))
    if phase.size > 1:
        phase = phase - phase[1]
    return phase


def run_phase_retrieval_ultrasound(
    waveform: np.ndarray,
    *,
    seed: int = 42,
    n_fft: int = 80,
    hop_length: int = 8,
    n_iter: int = 120,
) -> PhaseRetrievalResult:
    """Recover a real ultrasound waveform from its STFT magnitude only."""
    if n_fft < 16:
        raise ValueError("n_fft must be at least 16")
    if hop_length <= 0 or hop_length >= n_fft:
        raise ValueError("hop_length must be positive and smaller than n_fft")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")

    signal = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if signal.size < max(n_fft, 64):
        raise ValueError(f"waveform must contain at least {max(n_fft, 64)} samples")

    signal = signal - float(np.mean(signal))
    norm = float(np.linalg.norm(signal))
    if norm <= 1e-12:
        raise ValueError("waveform must contain non-zero energy")
    x_true = signal / norm

    measured_spectrogram = _stft_complex(x_true, n_fft=n_fft, hop_length=hop_length)
    measured_magnitude = np.abs(measured_spectrogram)
    initial_signal, recovered_signal, reconstructed_spectrogram, residual_curve = _griffin_lim_from_magnitude(
        measured_magnitude,
        signal_length=x_true.size,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        seed=seed,
    )
    _, init_error, _, _, _ = _best_aligned_signal(x_true, initial_signal)
    x_aligned, final_error, correlation, shift, scale = _best_aligned_signal(x_true, recovered_signal)
    true_phase = _phase_spectrum(x_true)
    recovered_phase = _phase_spectrum(x_aligned)
    phase_rmse = float(np.sqrt(np.mean((recovered_phase - true_phase) ** 2)))

    status = {
        "error_reduced": final_error < init_error,
        "final_error_below_0_35": final_error < 0.35,
        "correlation_above_0_9": correlation > 0.9,
    }
    status["overall_pass"] = bool(
        status["error_reduced"]
        and status["final_error_below_0_35"]
        and status["correlation_above_0_9"]
    )

    report = {
        "seed": int(seed),
        "solver": "griffin_lim",
        "measurement_model": "stft_magnitude_only",
        "signal_length": int(x_true.size),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "stft_rows": int(measured_spectrogram.shape[0]),
        "stft_cols": int(measured_spectrogram.shape[1]),
        "optimization_iterations": int(n_iter),
        "init_relative_error": init_error,
        "final_relative_error": float(final_error),
        "signal_correlation": float(correlation),
        "phase_rmse": phase_rmse,
        "alignment_shift": int(shift),
        "alignment_scale": float(scale),
        "residual_curve": [float(v) for v in residual_curve],
        "initial_consistency_error": float(residual_curve[0]),
        "final_consistency_error": float(residual_curve[-1]),
        "status": status,
    }
    return PhaseRetrievalResult(
        true_signal=x_true,
        recovered_signal=x_aligned,
        measured_spectrogram=measured_magnitude,
        reconstructed_spectrogram=np.abs(reconstructed_spectrogram),
        true_phase_spectrum=true_phase,
        recovered_phase_spectrum=recovered_phase,
        residual_curve=residual_curve,
        report=report,
        status=status,
    )


def run_phase_retrieval_transcranial(
    *,
    root_dir: str | None = None,
    case_name: str | None = None,
    window_length: int = 256,
    n_fft: int = 80,
    hop_length: int = 8,
    n_iter: int = 120,
    seed: int = 42,
    row_index: int | None = None,
    col_index: int | None = None,
    start_index: int | None = None,
) -> PhaseRetrievalResult:
    """Run STFT-magnitude phase retrieval on a real transcranial hydrophone pulse."""
    resolved_case = case_name or default_transcranial_case(root_dir)
    if start_index is None:
        window = select_high_energy_hydrophone_window(
            root_dir,
            case_name=resolved_case,
            window_length=window_length,
            row_index=row_index,
            col_index=col_index,
        )
    else:
        if row_index is None or col_index is None:
            raise ValueError("row_index and col_index are required when start_index is provided")
        window = load_transcranial_waveform_window(
            root_dir,
            case_name=resolved_case,
            row_index=row_index,
            col_index=col_index,
            start_index=start_index,
            window_length=window_length,
        )

    result = run_phase_retrieval_ultrasound(
        window.trace,
        seed=seed,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
    )
    signal_metadata = {
        "dataset": "ETH transcranial hydrophone scans",
        "case_name": window.case_name,
        "row_index": window.row_index,
        "col_index": window.col_index,
        "start_index": window.start_index,
        "window_length": window.window_length,
        "trace_energy": window.trace_energy,
        "dominant_frequency_bin": window.dominant_frequency_bin,
        "plane": window.case_name.rsplit("_", 1)[-1],
    }
    report = dict(result.report)
    report["dataset"] = signal_metadata
    return PhaseRetrievalResult(
        true_signal=result.true_signal,
        recovered_signal=result.recovered_signal,
        measured_spectrogram=result.measured_spectrogram,
        reconstructed_spectrogram=result.reconstructed_spectrogram,
        true_phase_spectrum=result.true_phase_spectrum,
        recovered_phase_spectrum=result.recovered_phase_spectrum,
        residual_curve=result.residual_curve,
        report=report,
        status=result.status,
        signal_metadata=signal_metadata,
        scan_energy_map=window.scan_energy_map,
    )


def tune_phase_retrieval_transcranial(
    *,
    root_dir: str | None = None,
    cases: Sequence[str] | None = None,
    window_lengths: Sequence[int] = (256,),
    n_fft_grid: Sequence[int] = (64, 80),
    hop_length_grid: Sequence[int] = (8, 12),
    iteration_grid: Sequence[int] = (80, 120),
    seed: int = 42,
) -> PhaseRetrievalTuningResult:
    """Grid-search tuned retrieval settings on locally available transcranial cases."""
    resolved_cases = list(cases) if cases is not None else list_transcranial_scan_cases(root_dir)
    if not resolved_cases:
        resolved_cases = [default_transcranial_case(root_dir)]

    ranked_results: list[dict[str, Any]] = []
    for window_length in window_lengths:
        for n_fft in n_fft_grid:
            for hop_length in hop_length_grid:
                if hop_length >= n_fft:
                    continue
                for n_iter in iteration_grid:
                    case_errors: dict[str, float] = {}
                    case_correlations: dict[str, float] = {}
                    for case_index, case_name in enumerate(resolved_cases):
                        result = run_phase_retrieval_transcranial(
                            root_dir=root_dir,
                            case_name=case_name,
                            window_length=int(window_length),
                            n_fft=int(n_fft),
                            hop_length=int(hop_length),
                            n_iter=int(n_iter),
                            seed=seed + case_index,
                        )
                        case_errors[case_name] = float(result.report["final_relative_error"])
                        case_correlations[case_name] = float(result.report["signal_correlation"])
                    ranked_results.append(
                        {
                            "window_length": int(window_length),
                            "n_fft": int(n_fft),
                            "hop_length": int(hop_length),
                            "n_iter": int(n_iter),
                            "solver": "griffin_lim",
                            "case_errors": case_errors,
                            "case_correlations": case_correlations,
                            "mean_final_relative_error": float(np.mean(list(case_errors.values()))),
                            "mean_signal_correlation": float(
                                np.mean(list(case_correlations.values()))
                            ),
                        }
                    )

    ranked_results.sort(
        key=lambda item: (
            item["mean_final_relative_error"],
            -item["mean_signal_correlation"],
        )
    )
    best = ranked_results[0] if ranked_results else {}
    return PhaseRetrievalTuningResult(
        best_config=best,
        ranked_results=ranked_results,
        cases=resolved_cases,
    )
