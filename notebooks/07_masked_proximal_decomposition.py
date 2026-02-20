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
# # 07. Masked Proximal Decomposition (A-Scan)
#
# ## Problem
# Ultrasound A-scan measurements can be partially missing or unreliable in some intervals.
# We want to decompose the signal into a smooth structural component and sparse reflections,
# while handling missing samples through a mask.
#
# ## Solution
# Solve a masked decomposition objective with alternating proximal-style updates:
# - Smooth component: quadratic solve with second-derivative regularization.
# - Sparse component: soft-threshold update.
# - Data consistency only on observed samples via mask `M`.
#
# ## Result
# The notebook returns reconstructed components, error curves, and a pass/fail status
# showing whether decomposition improves over a naive baseline.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_ndt_sample,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("07_masked_proximal_decomposition")
rng = np.random.default_rng(seed)

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")


# %%
def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def build_second_difference_matrix(n: int):
    """Second derivative finite-difference operator."""
    main = np.full(n, -2.0)
    off = np.ones(n - 1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    return D2


# %% [markdown]
# ## Build masked observation from local NDT sample

# %%
sample = load_ndt_sample("weld_inspection.npz")
rf = sample["rf"].astype(np.float64)

start = 180
length = 512
x_true = rf[start : start + length].copy()
x_true = x_true / (np.max(np.abs(x_true)) + 1e-12)

n = x_true.size
missing_rate = 0.22
mask = (rng.random(n) > missing_rate).astype(np.float64)

noise_std = 0.01
x_noisy = x_true + noise_std * rng.standard_normal(n)
y_obs = mask * x_noisy

print(f"n={n}, observed={int(mask.sum())}, missing={int((1-mask).sum())}")

# %% [markdown]
# ## Alternating masked decomposition

# %%
lam_smooth = 0.9
lam_sparse = 0.04
n_iter = 80

D2 = build_second_difference_matrix(n)
M = diags(mask, 0, format="csr")
I = eye(n, format="csr")
A_solve = M + lam_smooth * (D2.T @ D2) + 1e-6 * I

smooth = np.zeros(n)
sparse = np.zeros(n)
obj_hist: list[float] = []
rmse_hist: list[float] = []

for _ in range(n_iter):
    rhs = mask * (y_obs - sparse)
    smooth = np.asarray(spsolve(A_solve, rhs)).reshape(-1)

    residual_obs = mask * (y_obs - smooth)
    sparse = soft_threshold(residual_obs, lam_sparse)

    recon = smooth + sparse
    fit = 0.5 * np.sum((mask * (recon - y_obs)) ** 2)
    reg_s = 0.5 * lam_smooth * float(np.sum((D2 @ smooth) ** 2))
    reg_a = lam_sparse * float(np.sum(np.abs(sparse)))
    obj = fit + reg_s + reg_a
    obj_hist.append(obj)

    rmse_hist.append(float(np.sqrt(np.mean((recon - x_true) ** 2))))

x_hat = smooth + sparse

# %% [markdown]
# ## Quantitative result summary


# %%
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


baseline = y_obs  # zero-filled missing entries baseline
metrics = {
    "rmse_full": rmse(x_hat, x_true),
    "rmse_observed": rmse(x_hat[mask > 0], x_true[mask > 0]),
    "rmse_missing": rmse(x_hat[mask == 0], x_true[mask == 0]),
    "rmse_baseline_full": rmse(baseline, x_true),
    "objective_final": float(obj_hist[-1]),
}
metrics

# %% [markdown]
# ## Visual summary

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

x_idx = np.arange(n)
axes[0].plot(x_idx, x_true, color="#4C78A8", label="true")
axes[0].plot(x_idx, y_obs, color="#999999", alpha=0.7, label="observed (masked)")
axes[0].plot(x_idx, x_hat, color="#F58518", lw=1.2, label="reconstruction")
axes[0].set_title("Masked reconstruction")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(x_idx, smooth, color="#54A24B", label="smooth component")
axes[1].plot(x_idx, sparse, color="#B279A2", label="sparse component", alpha=0.9)
axes[1].set_title("Decomposed components")
axes[1].legend()
axes[1].grid(alpha=0.25)

axes[2].plot(obj_hist, color="#E45756", label="objective")
axes[2].plot(rmse_hist, color="#72B7B2", label="RMSE")
axes[2].set_title("Optimization traces")
axes[2].set_xlabel("Iteration")
axes[2].legend()
axes[2].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "masked_proximal_decomposition_summary.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Save report and status

# %%
status = {
    "improves_over_baseline": bool(metrics["rmse_full"] < metrics["rmse_baseline_full"]),
    "objective_decreases": bool(obj_hist[-1] < obj_hist[0]),
}
status["overall_pass"] = status["improves_over_baseline"] and status["objective_decreases"]

report = {
    "seed": seed,
    "sample": sample["name"],
    "n": int(n),
    "missing_rate": float(missing_rate),
    "lam_smooth": float(lam_smooth),
    "lam_sparse": float(lam_sparse),
    "iterations": int(n_iter),
    "metrics": metrics,
    "objective_history": [float(v) for v in obj_hist],
    "rmse_history": [float(v) for v in rmse_hist],
    "status": status,
}
save_json_report(output_dir / "masked_proximal_report.json", report)
status

# %% [markdown]
# ## Result interpretation
# - If `overall_pass` is `True`, decomposition improved reconstruction quality over zero-filled baseline.
# - If `overall_pass` is `False`, tune `lam_smooth`, `lam_sparse`, or mask assumptions.
# - This is a compact deterministic demonstration, not a full hyperparameter study.
