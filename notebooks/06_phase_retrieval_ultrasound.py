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
# Thin notebook wrapper around the reusable phase retrieval workflow.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_ndt_sample,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
from ultrasound.data.picmus_dataset import default_picmus_case, picmus_in_vivo_available
from ultrasound.workflows import run_phase_retrieval_picmus, run_phase_retrieval_ultrasound

seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("06_phase_retrieval_ultrasound")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
picmus_root = project_root / "data" / "picmus"
if picmus_in_vivo_available(picmus_root):
    case_name = default_picmus_case(picmus_root)
    result = run_phase_retrieval_picmus(
        root_dir=str(picmus_root),
        case_name=case_name,
        segment_length=96,
        measurement_ratio=5,
        n_iter=150,
        seed=seed,
    )
    report = dict(result.report)
    report["sample"] = f"PICMUS:{case_name}"
else:
    sample = load_ndt_sample("weld_inspection.npz")
    rf = sample["rf"].astype(float)
    start = 200
    length = 128
    rf_segment = rf[start : start + length]

    result = run_phase_retrieval_ultrasound(
        rf_segment,
        seed=seed,
        measurement_ratio=5,
        n_iter=150,
        solver="lbfgs",
    )
    report = dict(result.report)
    report["sample"] = sample["name"]
report

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(result.x_true.real, label="true", color="#4C78A8")
axes[0, 0].plot(result.x_aligned.real, label="recovered", color="#F58518", alpha=0.8)
axes[0, 0].set_title("Real part: true vs recovered")
axes[0, 0].grid(alpha=0.25)
axes[0, 0].legend()

axes[0, 1].plot(result.x_true.imag, label="true", color="#4C78A8")
axes[0, 1].plot(result.x_aligned.imag, label="recovered", color="#F58518", alpha=0.8)
axes[0, 1].set_title("Imaginary part: true vs recovered")
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend()

axes[1, 0].plot(result.amplitude_rmse, color="#54A24B")
axes[1, 0].set_title("Amplitude RMSE convergence")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("RMSE")
axes[1, 0].grid(alpha=0.25)

axes[1, 1].scatter(
    result.measured_amplitude,
    result.reconstructed_amplitude,
    s=8,
    alpha=0.6,
    color="#B279A2",
)
min_v = float(min(result.measured_amplitude.min(), result.reconstructed_amplitude.min()))
max_v = float(max(result.measured_amplitude.max(), result.reconstructed_amplitude.max()))
axes[1, 1].plot([min_v, max_v], [min_v, max_v], "k--", lw=1.0)
axes[1, 1].set_title("Measured vs reconstructed amplitudes")
axes[1, 1].set_xlabel("Measured |Ax|")
axes[1, 1].set_ylabel("Recovered |Ax_hat|")
axes[1, 1].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "phase_retrieval_summary.png", dpi=140)
plt.close(fig)

# %%
save_json_report(output_dir / "phase_retrieval_report.json", report)
result.status
