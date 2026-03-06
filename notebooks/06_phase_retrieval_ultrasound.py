# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06. Phase Retrieval for Ultrasound A-Scan
#
# ## Problem
# In many sensing chains we can reliably measure only magnitudes while phase is missing or corrupted.
# Without phase, signal reconstruction becomes ambiguous and downstream interpretation degrades.
#
# ## Solution
# Use a compact phase-retrieval workflow:
# 1. Build amplitude-only measurements from a local ultrasound A-scan segment.
# 2. Compute spectral initialization.
# 3. Refine with Wirtinger Flow iterations.
# 4. Evaluate reconstruction after global-phase alignment.
#
# ## Result
# This notebook reports initialization error vs final error, convergence behavior, and
# signal overlays for quick visual verification.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_ndt_sample,
    save_json_report,
    set_reproducible_seed,
)
from scipy.signal import hilbert

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("06_phase_retrieval_ultrasound")
rng = np.random.default_rng(seed)

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")


# %%
def align_global_phase(x_hat: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Align estimate to reference with a single global phase rotation."""
    theta = np.angle(np.vdot(x_true, x_hat))
    return x_hat * np.exp(-1j * theta)


def relative_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Relative reconstruction error after global-phase alignment."""
    aligned = align_global_phase(x_hat, x_true)
    return float(np.linalg.norm(aligned - x_true) / (np.linalg.norm(x_true) + 1e-12))


def spectral_init(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spectral initialization for phase retrieval."""
    m = A.shape[0]
    weighted = (y[:, None] ** 2) * A
    Y = (A.conj().T @ weighted) / m
    _, eigvecs = np.linalg.eigh(Y)
    v = eigvecs[:, -1]
    scale = np.sqrt(np.mean(y**2))
    return scale * v / (np.linalg.norm(v) + 1e-12)


def wirtinger_flow(
    A: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    n_iter: int = 180,
    step_size: float = 0.25,
) -> tuple[np.ndarray, list[float]]:
    """Amplitude-based Wirtinger Flow with fixed step size."""
    m = A.shape[0]
    x = x0.copy()
    amp_errors: list[float] = []

    for _ in range(n_iter):
        Ax = A @ x
        amp = np.abs(Ax)
        residual = (amp - y) * (Ax / (amp + 1e-12))
        grad = (A.conj().T @ residual) / m
        x = x - step_size * grad

        amp_rmse = float(np.sqrt(np.mean((np.abs(A @ x) - y) ** 2)))
        amp_errors.append(amp_rmse)

    return x, amp_errors


# %% [markdown]
# ## Build a local phase-retrieval test instance

# %%
sample = load_ndt_sample("weld_inspection.npz")
rf = sample["rf"].astype(np.float64)

# Keep a compact segment for fast experimentation.
start = 200
length = 256
rf_segment = rf[start : start + length]

analytic = hilbert(rf_segment)
x_true = analytic / (np.linalg.norm(analytic) + 1e-12)

n = x_true.size
m = 4 * n
A = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(m)

y_clean = np.abs(A @ x_true)
noise = 0.01 * np.median(y_clean) * rng.standard_normal(m)
y = np.clip(y_clean + noise, 1e-10, None)

print(f"Signal length n={n}, measurements m={m}")

# %% [markdown]
# ## Initialization and iterative recovery

# %%
x0 = spectral_init(A, y)
init_error = relative_error(x0, x_true)

x_hat, amp_errors = wirtinger_flow(A, y, x0=x0, n_iter=180, step_size=0.22)
final_error = relative_error(x_hat, x_true)

x_aligned = align_global_phase(x_hat, x_true)

summary = {
    "init_relative_error": init_error,
    "final_relative_error": final_error,
    "amp_rmse_final": float(amp_errors[-1]),
}
summary

# %% [markdown]
# ## Visual analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(np.real(x_true), label="true", color="#4C78A8")
axes[0, 0].plot(np.real(x_aligned), label="recovered", color="#F58518", alpha=0.8)
axes[0, 0].set_title("Real part: true vs recovered")
axes[0, 0].grid(alpha=0.25)
axes[0, 0].legend()

axes[0, 1].plot(np.imag(x_true), label="true", color="#4C78A8")
axes[0, 1].plot(np.imag(x_aligned), label="recovered", color="#F58518", alpha=0.8)
axes[0, 1].set_title("Imaginary part: true vs recovered")
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend()

axes[1, 0].plot(amp_errors, color="#54A24B")
axes[1, 0].set_title("Amplitude RMSE convergence")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("RMSE")
axes[1, 0].grid(alpha=0.25)

pred_amp = np.abs(A @ x_aligned)
axes[1, 1].scatter(y, pred_amp, s=8, alpha=0.6, color="#B279A2")
min_v = float(min(y.min(), pred_amp.min()))
max_v = float(max(y.max(), pred_amp.max()))
axes[1, 1].plot([min_v, max_v], [min_v, max_v], "k--", lw=1.0)
axes[1, 1].set_title("Measured vs reconstructed amplitudes")
axes[1, 1].set_xlabel("Measured |Ax|")
axes[1, 1].set_ylabel("Recovered |Ax_hat|")
axes[1, 1].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "phase_retrieval_summary.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Save report and status

# %%
status = {
    "error_reduced": final_error < init_error,
    "final_error_below_0_5": final_error < 0.5,
}
status["overall_pass"] = status["error_reduced"] and status["final_error_below_0_5"]

report = {
    "seed": seed,
    "sample": sample["name"],
    "n": int(n),
    "m": int(m),
    "init_relative_error": float(init_error),
    "final_relative_error": float(final_error),
    "amplitude_rmse_curve": [float(v) for v in amp_errors],
    "status": status,
}
save_json_report(output_dir / "phase_retrieval_report.json", report)
status

# %% [markdown]
# ## Result interpretation
# - If `overall_pass` is `True`, the solver improved reconstruction from the spectral initialization.
# - If `overall_pass` is `False`, tune `step_size`, `n_iter`, or measurement ratio `m/n`.
# - This is a compact demonstration, not a production-grade phase retrieval benchmark.
