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
# # 05. NDT A-Scan Analysis
# Thin notebook wrapper around the reusable NDT A-scan workflow.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_ndt_sample,
    save_json_report,
    set_reproducible_seed,
    summarize_ndt_samples,
)

project_root = ensure_src_on_path()
from ultrasound.workflows import run_ndt_ascan_analysis

seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("05_ndt_ascan_analysis")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
rows = summarize_ndt_samples()
rows

# %%
sample_name = "steel_plate_with_crack.npz"
sample = load_ndt_sample(sample_name)
result = run_ndt_ascan_analysis(
    sample["rf"],
    sample["time"],
    fs_hz=float(sample["fs"]),
    fc_hz=float(sample["fc"]),
    c_mps=float(sample["c"]),
    nominal_thickness_m=float(sample["thickness"]),
    seed=seed,
)
report = dict(result.report)
report["sample"] = sample_name
report

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
axes[0].plot(sample["time"] * 1e6, sample["rf"], color="#4C78A8", lw=1.0)
axes[0].set_title("A-scan RF signal")
axes[0].set_xlabel("Time [us]")
axes[0].set_ylabel("Amplitude")
axes[0].grid(alpha=0.25)

axes[1].plot(sample["time"] * 1e6, result.envelope_db, color="#F58518", lw=1.2)
axes[1].scatter(
    sample["time"][result.peak_indices] * 1e6,
    result.envelope_db[result.peak_indices],
    color="crimson",
    s=24,
    zorder=3,
)
axes[1].set_title("Envelope (dB) and detected echoes")
axes[1].set_xlabel("Time [us]")
axes[1].set_ylabel("Magnitude [dB]")
axes[1].set_ylim(-80, 5)
axes[1].grid(alpha=0.25)

axes[2].plot(result.freq_hz / 1e6, result.spectrum_db, color="#54A24B", lw=1.2)
axes[2].axvline(float(sample["fc"]) / 1e6, color="black", ls="--", lw=1.0, label="Nominal fc")
axes[2].set_title("Frequency spectrum")
axes[2].set_xlabel("Frequency [MHz]")
axes[2].set_ylabel("Magnitude [dB]")
axes[2].set_xlim(0, min(20.0, float(sample["fs"]) / 2e6))
axes[2].set_ylim(-80, 5)
axes[2].grid(alpha=0.25)
axes[2].legend()

plt.tight_layout()
fig.savefig(output_dir / "ascan_summary.png", dpi=140)
plt.close(fig)

# %%
save_json_report(output_dir / "ndt_ascan_report.json", report)
result.status
