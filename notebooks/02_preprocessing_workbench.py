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
# # 02. Preprocessing Workbench
#
# Focused notebook for validating image preprocessing algorithms in `src/ultrasound/preprocessing`.
#
# ## Problem
# Raw ultrasound images suffer from speckle noise and low contrast.
# If preprocessing is too weak, structures remain unclear; if too strong, lesion boundaries are destroyed.
#
# ## Solution
# Apply and compare multiple preprocessing strategies on the same input:
# - Speckle filters (Lee, Frost, Median)
# - Contrast enhancement (CLAHE)
# - Optimization-based denoising (ADMM-TV)
#
# ## Result
# The notebook generates side-by-side visuals and quantitative metrics so we can judge
# whether noise suppression improved without unacceptable structure loss.
#
# ## Scope
# This notebook is for algorithm behavior checks, not benchmark ranking.
# It compares classical filters and ADMM-TV on one representative BUSI image
# and saves both visuals and numeric summaries.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_busi_sample_arrays,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("02_preprocessing_workbench")

from ultrasound.utils.visualization import plot_preprocessing_comparison
from ultrasound.workflows import run_preprocessing_workbench

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Load one BUSI sample

# %%
image_rgb, _ = load_busi_sample_arrays(class_name="malignant")
result = run_preprocessing_workbench(image_rgb)
gray = result.gray

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image_rgb)
axes[0].set_title("RGB input")
axes[0].axis("off")
axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Grayscale")
axes[1].axis("off")
plt.tight_layout()
fig.savefig(output_dir / "input_sample.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Speckle reduction + contrast enhancement + ADMM-TV denoising

# %%
processed = result.processed
img_lee = processed["Lee"]
img_frost = processed["Frost"]
img_tv = processed["ADMM-TV"]
convergence = result.convergence

fig = plot_preprocessing_comparison(gray, processed, figsize=(18, 4))
fig.savefig(output_dir / "preprocessing_comparison.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Quantitative checks
#
# Metric meaning in this context:
# - RMSE: lower means less per-pixel deviation from the reference.
# - PSNR: higher means closer to the reference intensity distribution.
# - SSIM: higher means better structural similarity.
# - Speckle CV: lower usually indicates stronger smoothing.

# %%
quality = result.quality
mean_val = result.mean_intensity
cv_before = result.speckle_cv["original"]
cv_after_lee = result.speckle_cv["lee"]
cv_after_frost = result.speckle_cv["frost"]
cv_after_tv = result.speckle_cv["tv"]

quality

# %% [markdown]
# Interpretation note:
# - No single method is universally best for ultrasound.
# - If CV drops strongly but lesion boundaries blur, smoothing is too aggressive.
# - Prefer settings that reduce speckle while preserving lesion contours and edge transitions.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["original", "lee", "frost", "tv"]
values = [cv_before, cv_after_lee, cv_after_frost, cv_after_tv]
ax.plot(labels, values, marker="o")
ax.set_title("Speckle coefficient of variation")
ax.set_ylabel("CV")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(output_dir / "speckle_cv_comparison.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Persist run report

# %%
report = {
    "seed": seed,
    "image_shape": list(image_rgb.shape),
    "mean_intensity": float(mean_val),
    "cv": {
        "original": float(cv_before),
        "lee": float(cv_after_lee),
        "frost": float(cv_after_frost),
        "tv": float(cv_after_tv),
    },
    "quality": quality,
    "admm": {
        "iterations": int(len(convergence["primal_residuals"])),
        "final_primal": float(convergence["primal_residuals"][-1]),
        "final_dual": float(convergence["dual_residuals"][-1]),
    },
}

save_json_report(output_dir / "preprocessing_report.json", report)
report

# %%
result_summary = {
    "best_psnr_method": max(quality, key=lambda k: quality[k]["psnr"]),
    "best_ssim_method": max(quality, key=lambda k: quality[k]["ssim"]),
    "lowest_cv_method": min(
        {"lee": cv_after_lee, "frost": cv_after_frost, "tv": cv_after_tv},
        key=lambda k: {"lee": cv_after_lee, "frost": cv_after_frost, "tv": cv_after_tv}[k],
    ),
}
result_summary

# %% [markdown]
# ## What to do next
# - If ADMM residuals stagnate, tune `lambda_tv` and `rho`.
# - If edges are over-smoothed, reduce filter window size or regularization strength.
# - For robust conclusions, repeat this workflow on a stratified sample set.
#
# ## Result interpretation
# - Use `best_ssim_method` as the primary structural-preservation hint.
# - Use `lowest_cv_method` as the strongest speckle-suppression hint.
# - Prefer the method that balances both, not the one that maximizes only one metric.
