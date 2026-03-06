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
#
# Compact workflow for local NDT A-scan ultrasound data.
# Uses the packaged `.npz` signals under `data/ascan_signals/ndt_samples`.
#
# ## Problem
# A-scan signals are hard to read directly, making defect localization and thickness estimation unreliable
# without signal-domain analysis (envelope, peak detection, and spectrum review).
#
# ## Solution
# Build a compact analysis flow:
# - Envelope extraction with Hilbert transform.
# - Echo peak picking with configurable thresholds.
# - Thickness estimate from time-of-flight.
# - Spectrum inspection around nominal center frequency.
#
# ## Result
# The notebook provides detectable echo timings, thickness estimate/error, and summary plots,
# plus a JSON report suitable for traceability.
#
# ## Assumptions
# - Pulse-echo geometry.
# - Approximate constant wave speed `c` provided by metadata.
# - Echo peak picking is threshold-based and therefore sensitive to SNR.

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
    summarize_ndt_samples,
)
from scipy.signal import find_peaks, hilbert

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("05_ndt_ascan_analysis")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Select sample and inspect metadata

# %%
rows = summarize_ndt_samples()
rows

# %%
sample_name = "steel_plate_with_crack.npz"
sample = load_ndt_sample(sample_name)

rf = sample["rf"]
time_s = sample["time"]
fs = float(sample["fs"])
fc = float(sample["fc"])
c_mps = float(sample["c"])

print(f"Sample: {sample_name}")
print(f"n_points: {rf.size}")
print(f"fs: {fs/1e6:.2f} MHz, fc: {fc/1e6:.2f} MHz, c: {c_mps:.1f} m/s")
print(f"Nominal thickness: {sample['thickness']*1e3:.2f} mm")
print(f"Defects: {sample['defects']}")

# %% [markdown]
# ## Envelope and echo detection

# %%
envelope = np.abs(hilbert(rf))
envelope_db = 20.0 * np.log10(envelope / (np.max(envelope) + 1e-12) + 1e-12)

peak_threshold = 0.2 * np.max(envelope)
min_dist = max(10, int(0.4e-6 * fs))
peaks, properties = find_peaks(envelope, height=peak_threshold, distance=min_dist)

print(f"Detected peaks: {len(peaks)}")
print("Peak times [us]:", (time_s[peaks] * 1e6).round(3).tolist())

# %% [markdown]
# Interpretation note:
# - If too many peaks are detected, increase `peak_threshold` or `min_dist`.
# - If expected echoes are missing, reduce the threshold or inspect signal conditioning.

# %% [markdown]
# ## Thickness estimate from front/back wall echoes

# %%
estimated_thickness_mm = float("nan")
if len(peaks) >= 2:
    # Use the first two prominent echoes as a simple pulse-echo estimator.
    tof = float(time_s[peaks[1]] - time_s[peaks[0]])
    estimated_thickness_mm = 0.5 * c_mps * tof * 1e3

estimated_thickness_mm

# %%
nominal_thickness_mm = float(sample["thickness"] * 1e3)
thickness_error_mm = float(estimated_thickness_mm - nominal_thickness_mm)
thickness_error_mm

# %% [markdown]
# Interpretation note:
# - Positive error means over-estimated thickness.
# - Negative error means under-estimated thickness.
# - For production, replace this simple two-peak method with robust front/back wall tracking.

# %% [markdown]
# ## Frequency-domain analysis

# %%
n = rf.size
window = np.hanning(n)
rf_win = rf * window
spectrum = np.fft.rfft(rf_win)
freq_hz = np.fft.rfftfreq(n, d=1.0 / fs)
mag_db = 20.0 * np.log10(np.abs(spectrum) / (np.max(np.abs(spectrum)) + 1e-12) + 1e-12)

# %% [markdown]
# ## Visual summary

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(time_s * 1e6, rf, color="#4C78A8", lw=1.0)
axes[0].set_title("A-scan RF signal")
axes[0].set_xlabel("Time [us]")
axes[0].set_ylabel("Amplitude")
axes[0].grid(alpha=0.25)

axes[1].plot(time_s * 1e6, envelope_db, color="#F58518", lw=1.2)
axes[1].scatter(time_s[peaks] * 1e6, envelope_db[peaks], color="crimson", s=24, zorder=3)
axes[1].set_title("Envelope (dB) and detected echoes")
axes[1].set_xlabel("Time [us]")
axes[1].set_ylabel("Magnitude [dB]")
axes[1].set_ylim(-80, 5)
axes[1].grid(alpha=0.25)

axes[2].plot(freq_hz / 1e6, mag_db, color="#54A24B", lw=1.2)
axes[2].axvline(fc / 1e6, color="black", ls="--", lw=1.0, label="Nominal fc")
axes[2].set_title("Frequency spectrum")
axes[2].set_xlabel("Frequency [MHz]")
axes[2].set_ylabel("Magnitude [dB]")
axes[2].set_xlim(0, min(20.0, fs / 2e6))
axes[2].set_ylim(-80, 5)
axes[2].grid(alpha=0.25)
axes[2].legend()

plt.tight_layout()
fig.savefig(output_dir / "ascan_summary.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Save analysis report

# %%
report = {
    "seed": seed,
    "sample": sample_name,
    "fs_hz": fs,
    "fc_hz": fc,
    "c_mps": c_mps,
    "n_points": int(n),
    "detected_echoes": int(len(peaks)),
    "peak_times_us": (time_s[peaks] * 1e6).round(6).tolist(),
    "nominal_thickness_mm": nominal_thickness_mm,
    "estimated_thickness_mm": float(estimated_thickness_mm),
    "thickness_error_mm": thickness_error_mm,
}
save_json_report(output_dir / "ndt_ascan_report.json", report)
report

# %%
analysis_status = {
    "echoes_detected": int(len(peaks)),
    "thickness_estimate_available": bool(np.isfinite(estimated_thickness_mm)),
    "abs_thickness_error_mm": (
        float(abs(thickness_error_mm)) if np.isfinite(thickness_error_mm) else float("nan")
    ),
}
analysis_status["overall_pass"] = (
    analysis_status["echoes_detected"] >= 2 and analysis_status["thickness_estimate_available"]
)
analysis_status

# %% [markdown]
# ## Result interpretation
# - If `overall_pass` is `True`, the basic A-scan analysis workflow produced usable outputs.
# - `abs_thickness_error_mm` provides quick quality feedback for this simple estimator.
# - For production use, prefer robust wall-tracking and uncertainty-aware peak selection.
