"""Reusable phase retrieval workflow for real ultrasound RF segments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.signal import hilbert

from ultrasound.data.picmus_dataset import (
    default_picmus_case,
    list_picmus_rf_cases,
    load_picmus_rf_segment,
    select_high_energy_rf_segment,
)


@dataclass(frozen=True)
class PhaseRetrievalResult:
    x_true: np.ndarray
    x_aligned: np.ndarray
    measured_amplitude: np.ndarray
    reconstructed_amplitude: np.ndarray
    amplitude_rmse: list[float]
    report: dict[str, Any]
    status: dict[str, bool]
    signal_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PhaseRetrievalTuningResult:
    best_config: dict[str, Any]
    ranked_results: list[dict[str, Any]]
    cases: list[str]


def align_global_phase(x_hat: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    theta = np.angle(np.vdot(x_true, x_hat))
    return x_hat * np.exp(-1j * theta)


def relative_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    aligned = align_global_phase(x_hat, x_true)
    return float(np.linalg.norm(aligned - x_true) / (np.linalg.norm(x_true) + 1e-12))


def spectral_init(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = A.shape[0]
    weighted = (y[:, None] ** 2) * A
    gram = (A.conj().T @ weighted) / m
    _, eigvecs = np.linalg.eigh(gram)
    v = eigvecs[:, -1]
    scale = np.sqrt(np.mean(y**2))
    return scale * v / (np.linalg.norm(v) + 1e-12)


def wirtinger_flow(
    A: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    *,
    n_iter: int = 180,
    step_size: float = 0.25,
) -> tuple[np.ndarray, list[float]]:
    m = A.shape[0]
    x = x0.copy()
    amp_errors: list[float] = []

    for _ in range(int(n_iter)):
        Ax = A @ x
        amp = np.abs(Ax)
        residual = (amp - y) * (Ax / (amp + 1e-12))
        grad = (A.conj().T @ residual) / m
        x = x - float(step_size) * grad
        amp_errors.append(float(np.sqrt(np.mean((np.abs(A @ x) - y) ** 2))))

    return x, amp_errors


def _complex_gaussian_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    return (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0 * m)


def _complex_to_real_vector(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.real, x.imag])


def _real_vector_to_complex(z: np.ndarray, n: int) -> np.ndarray:
    return z[:n] + 1j * z[n:]


def _amplitude_rmse(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.abs(A @ x) - y) ** 2)))


def _amplitude_loss_and_grad(z: np.ndarray, A: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    n = A.shape[1]
    x = _real_vector_to_complex(z, n)
    Ax = A @ x
    amp = np.abs(Ax)
    residual = amp - y
    loss = 0.5 * np.mean(residual**2)
    grad_complex = (A.conj().T @ (residual * (Ax / np.maximum(amp, 1e-12)))) / A.shape[0]
    grad = _complex_to_real_vector(grad_complex)
    return float(loss), grad


def lbfgs_phase_retrieval(
    A: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    *,
    max_iterations: int = 150,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    n = A.shape[1]
    history = [_amplitude_rmse(A, x0, y)]

    def callback(z: np.ndarray) -> None:
        x = _real_vector_to_complex(z, n)
        history.append(_amplitude_rmse(A, x, y))

    result = minimize(
        _amplitude_loss_and_grad,
        _complex_to_real_vector(x0),
        args=(A, y),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": int(max_iterations), "maxls": 30},
    )
    x_hat = _real_vector_to_complex(np.asarray(result.x, dtype=np.float64), n)
    final_rmse = _amplitude_rmse(A, x_hat, y)
    if not history or abs(history[-1] - final_rmse) > 1e-12:
        history.append(final_rmse)
    return x_hat, history, result


def run_phase_retrieval_ultrasound(
    rf_segment: np.ndarray,
    *,
    seed: int = 42,
    measurement_ratio: int = 5,
    noise_scale: float = 0.0,
    n_iter: int = 150,
    step_size: float = 0.22,
    solver: Literal["lbfgs", "wirtinger"] = "lbfgs",
) -> PhaseRetrievalResult:
    if measurement_ratio <= 0:
        raise ValueError("measurement_ratio must be positive")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")

    rng = np.random.default_rng(seed)
    real_segment = np.asarray(rf_segment, dtype=np.float64).reshape(-1)
    if real_segment.size < 32:
        raise ValueError("rf_segment must contain at least 32 samples")

    analytic = hilbert(real_segment)
    x_true = analytic / (np.linalg.norm(analytic) + 1e-12)

    n = int(x_true.size)
    m = int(measurement_ratio * n)
    A = _complex_gaussian_matrix(rng, m, n)

    y_clean = np.abs(A @ x_true)
    if noise_scale > 0.0:
        noise = float(noise_scale) * float(np.median(y_clean)) * rng.standard_normal(m)
        y = np.clip(y_clean + noise, 1e-10, None)
    else:
        y = y_clean

    x0 = spectral_init(A, y)
    init_error = relative_error(x0, x_true)

    optimization: OptimizeResult | None = None
    if solver == "lbfgs":
        x_hat, amp_errors, optimization = lbfgs_phase_retrieval(A, y, x0, max_iterations=n_iter)
    else:
        x_hat, amp_errors = wirtinger_flow(A, y, x0, n_iter=n_iter, step_size=step_size)

    final_error = relative_error(x_hat, x_true)
    x_aligned = align_global_phase(x_hat, x_true)

    status = {
        "error_reduced": final_error < init_error,
        "final_error_below_0_25": final_error < 0.25,
    }
    status["overall_pass"] = bool(status["error_reduced"] and status["final_error_below_0_25"])

    report = {
        "seed": int(seed),
        "solver": solver,
        "operator": "complex_gaussian_amplitude_only",
        "n": n,
        "m": m,
        "measurement_ratio": int(measurement_ratio),
        "init_relative_error": float(init_error),
        "final_relative_error": float(final_error),
        "amplitude_rmse_curve": [float(v) for v in amp_errors],
        "optimization_iterations": int(getattr(optimization, "nit", len(amp_errors))),
        "optimization_success": bool(getattr(optimization, "success", True)),
        "optimization_message": str(getattr(optimization, "message", "")),
        "status": status,
    }
    return PhaseRetrievalResult(
        x_true=x_true,
        x_aligned=x_aligned,
        measured_amplitude=y,
        reconstructed_amplitude=np.abs(A @ x_aligned),
        amplitude_rmse=amp_errors,
        report=report,
        status=status,
    )


def run_phase_retrieval_picmus(
    *,
    root_dir: str | None = None,
    case_name: str | None = None,
    segment_length: int = 96,
    measurement_ratio: int = 5,
    n_iter: int = 150,
    seed: int = 42,
    noise_scale: float = 0.0,
    solver: Literal["lbfgs", "wirtinger"] = "lbfgs",
    angle_index: int | None = None,
    element_index: int | None = None,
    start_index: int | None = None,
) -> PhaseRetrievalResult:
    """Run tuned phase retrieval on a real PICMUS RF segment."""
    resolved_case = case_name or default_picmus_case(root_dir)
    if start_index is None:
        segment = select_high_energy_rf_segment(
            root_dir,
            case_name=resolved_case,
            segment_length=segment_length,
            angle_index=angle_index,
            element_index=element_index,
        )
    else:
        if angle_index is None or element_index is None:
            raise ValueError(
                "angle_index and element_index are required when start_index is provided"
            )
        segment = load_picmus_rf_segment(
            root_dir,
            case_name=resolved_case,
            angle_index=angle_index,
            element_index=element_index,
            start_index=start_index,
            segment_length=segment_length,
        )

    result = run_phase_retrieval_ultrasound(
        segment.rf_segment,
        seed=seed,
        measurement_ratio=measurement_ratio,
        noise_scale=noise_scale,
        n_iter=n_iter,
        solver=solver,
    )
    report = dict(result.report)
    signal_metadata = {
        "dataset": "PICMUS in_vivo",
        "case_name": segment.case_name,
        "angle_index": segment.angle_index,
        "element_index": segment.element_index,
        "start_index": segment.start_index,
        "segment_length": segment.segment_length,
        "energy": segment.energy,
        "sampling_frequency_hz": segment.sampling_frequency_hz,
        "sound_speed_mps": segment.sound_speed_mps,
    }
    report["dataset"] = signal_metadata
    return PhaseRetrievalResult(
        x_true=result.x_true,
        x_aligned=result.x_aligned,
        measured_amplitude=result.measured_amplitude,
        reconstructed_amplitude=result.reconstructed_amplitude,
        amplitude_rmse=result.amplitude_rmse,
        report=report,
        status=result.status,
        signal_metadata=signal_metadata,
    )


def tune_phase_retrieval_picmus(
    *,
    root_dir: str | None = None,
    cases: Sequence[str] | None = None,
    segment_lengths: Sequence[int] = (96, 128),
    measurement_ratios: Sequence[int] = (4, 5, 6),
    iteration_grid: Sequence[int] = (150, 300),
    solver: Literal["lbfgs", "wirtinger"] = "lbfgs",
    seed: int = 42,
) -> PhaseRetrievalTuningResult:
    """Grid-search tuned retrieval settings on locally available PICMUS cases."""
    resolved_cases = list(cases) if cases is not None else list_picmus_rf_cases(root_dir)
    if not resolved_cases:
        resolved_cases = [default_picmus_case(root_dir)]

    ranked_results: list[dict[str, Any]] = []
    for segment_length in segment_lengths:
        for measurement_ratio in measurement_ratios:
            for n_iter in iteration_grid:
                case_errors: dict[str, float] = {}
                for case_index, case_name in enumerate(resolved_cases):
                    result = run_phase_retrieval_picmus(
                        root_dir=root_dir,
                        case_name=case_name,
                        segment_length=int(segment_length),
                        measurement_ratio=int(measurement_ratio),
                        n_iter=int(n_iter),
                        solver=solver,
                        seed=seed + case_index,
                    )
                    case_errors[case_name] = float(result.report["final_relative_error"])
                ranked_results.append(
                    {
                        "segment_length": int(segment_length),
                        "measurement_ratio": int(measurement_ratio),
                        "n_iter": int(n_iter),
                        "solver": solver,
                        "case_errors": case_errors,
                        "mean_final_relative_error": float(np.mean(list(case_errors.values()))),
                    }
                )

    ranked_results.sort(key=lambda item: item["mean_final_relative_error"])
    best = ranked_results[0] if ranked_results else {}
    return PhaseRetrievalTuningResult(
        best_config=best,
        ranked_results=ranked_results,
        cases=resolved_cases,
    )
