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
# Thin notebook wrapper around the reusable masked proximal decomposition workflow.

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

project_root = ensure_src_on_path()
from ultrasound.workflows import run_masked_proximal_decomposition

seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("07_masked_proximal_decomposition")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
sample = load_ndt_sample("weld_inspection.npz")
rf = sample["rf"].astype(np.float64)
start = 180
length = 512
signal = rf[start : start + length]

result = run_masked_proximal_decomposition(
    signal,
    seed=seed,
    missing_rate=0.22,
    noise_std=0.01,
    lam_smooth=0.9,
    lam_sparse=0.04,
    n_iter=80,
)
report = dict(result.report)
report["sample"] = sample["name"]
report

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
x_idx = np.arange(result.reconstruction.size)
axes[0].plot(x_idx, signal / (np.max(np.abs(signal)) + 1e-12), color="#4C78A8", label="true")
axes[0].plot(x_idx, result.observed, color="#999999", alpha=0.7, label="observed (masked)")
axes[0].plot(x_idx, result.reconstruction, color="#F58518", lw=1.2, label="reconstruction")
axes[0].set_title("Masked reconstruction")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(x_idx, result.smooth, color="#54A24B", label="smooth component")
axes[1].plot(x_idx, result.sparse, color="#B279A2", label="sparse component", alpha=0.9)
axes[1].set_title("Decomposed components")
axes[1].legend()
axes[1].grid(alpha=0.25)

axes[2].plot(result.objective_history, color="#E45756", label="objective")
axes[2].plot(result.rmse_history, color="#72B7B2", label="RMSE")
axes[2].set_title("Optimization traces")
axes[2].set_xlabel("Iteration")
axes[2].legend()
axes[2].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "masked_proximal_decomposition_summary.png", dpi=140)
plt.close(fig)

# %%
save_json_report(output_dir / "masked_proximal_report.json", report)
result.status
