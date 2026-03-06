"""Reusable masked proximal decomposition workflow for A-scan signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class MaskedProximalDecompositionResult:
    observed: np.ndarray
    reconstruction: np.ndarray
    smooth: np.ndarray
    sparse: np.ndarray
    mask: np.ndarray
    objective_history: list[float]
    rmse_history: list[float]
    report: dict[str, Any]
    status: dict[str, bool]


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def build_second_difference_matrix(n: int):
    main = np.full(n, -2.0)
    off = np.ones(n - 1)
    return diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_masked_proximal_decomposition(
    signal: np.ndarray,
    *,
    seed: int = 42,
    missing_rate: float = 0.22,
    noise_std: float = 0.01,
    lam_smooth: float = 0.9,
    lam_sparse: float = 0.04,
    n_iter: int = 80,
) -> MaskedProximalDecompositionResult:
    rng = np.random.default_rng(seed)

    x_true = np.asarray(signal, dtype=np.float64).reshape(-1).copy()
    x_true = x_true / (np.max(np.abs(x_true)) + 1e-12)

    n = int(x_true.size)
    mask = (rng.random(n) > float(missing_rate)).astype(np.float64)
    x_noisy = x_true + float(noise_std) * rng.standard_normal(n)
    y_obs = mask * x_noisy

    D2 = build_second_difference_matrix(n)
    M = diags(mask, 0, format="csr")
    identity = eye(n, format="csr")
    A_solve = M + float(lam_smooth) * (D2.T @ D2) + 1e-6 * identity

    smooth = np.zeros(n)
    sparse = np.zeros(n)
    obj_hist: list[float] = []
    rmse_hist: list[float] = []

    for _ in range(int(n_iter)):
        rhs = mask * (y_obs - sparse)
        smooth = np.asarray(spsolve(A_solve, rhs)).reshape(-1)

        residual_obs = mask * (y_obs - smooth)
        sparse = soft_threshold(residual_obs, float(lam_sparse))

        recon = smooth + sparse
        fit = 0.5 * np.sum((mask * (recon - y_obs)) ** 2)
        reg_s = 0.5 * float(lam_smooth) * float(np.sum((D2 @ smooth) ** 2))
        reg_a = float(lam_sparse) * float(np.sum(np.abs(sparse)))
        obj_hist.append(float(fit + reg_s + reg_a))
        rmse_hist.append(_rmse(recon, x_true))

    x_hat = smooth + sparse
    metrics = {
        "rmse_full": _rmse(x_hat, x_true),
        "rmse_observed": _rmse(x_hat[mask > 0], x_true[mask > 0]),
        "rmse_missing": _rmse(x_hat[mask == 0], x_true[mask == 0]),
        "rmse_baseline_full": _rmse(y_obs, x_true),
        "objective_final": float(obj_hist[-1]),
    }
    status = {
        "improves_over_baseline": bool(metrics["rmse_full"] < metrics["rmse_baseline_full"]),
        "objective_decreases": bool(obj_hist[-1] <= (obj_hist[0] + 1e-9)),
    }
    status["overall_pass"] = bool(
        status["improves_over_baseline"] and status["objective_decreases"]
    )

    report = {
        "seed": int(seed),
        "n": n,
        "missing_rate": float(missing_rate),
        "lam_smooth": float(lam_smooth),
        "lam_sparse": float(lam_sparse),
        "iterations": int(n_iter),
        "metrics": metrics,
        "objective_history": [float(v) for v in obj_hist],
        "rmse_history": [float(v) for v in rmse_hist],
        "status": status,
    }
    return MaskedProximalDecompositionResult(
        observed=y_obs,
        reconstruction=x_hat,
        smooth=smooth,
        sparse=sparse,
        mask=mask,
        objective_history=obj_hist,
        rmse_history=rmse_hist,
        report=report,
        status=status,
    )
