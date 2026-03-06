"""Reusable phase retrieval workflow for ultrasound A-scan segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class PhaseRetrievalResult:
    x_true: np.ndarray
    x_aligned: np.ndarray
    measured_amplitude: np.ndarray
    reconstructed_amplitude: np.ndarray
    amplitude_rmse: list[float]
    report: dict[str, Any]
    status: dict[str, bool]


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


def run_phase_retrieval_ultrasound(
    rf_segment: np.ndarray,
    *,
    seed: int = 42,
    measurement_ratio: int = 4,
    noise_scale: float = 0.01,
    n_iter: int = 180,
    step_size: float = 0.22,
) -> PhaseRetrievalResult:
    rng = np.random.default_rng(seed)

    analytic = hilbert(np.asarray(rf_segment, dtype=np.float64).reshape(-1))
    x_true = analytic / (np.linalg.norm(analytic) + 1e-12)

    n = int(x_true.size)
    m = int(measurement_ratio * n)
    A = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(m)

    y_clean = np.abs(A @ x_true)
    noise = float(noise_scale) * float(np.median(y_clean)) * rng.standard_normal(m)
    y = np.clip(y_clean + noise, 1e-10, None)

    x0 = spectral_init(A, y)
    init_error = relative_error(x0, x_true)
    x_hat, amp_errors = wirtinger_flow(A, y, x0, n_iter=n_iter, step_size=step_size)
    final_error = relative_error(x_hat, x_true)
    x_aligned = align_global_phase(x_hat, x_true)

    status = {
        "error_reduced": final_error < init_error,
        "final_error_below_0_5": final_error < 0.5,
    }
    status["overall_pass"] = bool(status["error_reduced"] and status["final_error_below_0_5"])

    report = {
        "seed": int(seed),
        "n": n,
        "m": m,
        "init_relative_error": float(init_error),
        "final_relative_error": float(final_error),
        "amplitude_rmse_curve": [float(v) for v in amp_errors],
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
