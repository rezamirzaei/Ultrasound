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
# # 01. Dataset Healthcheck
#
# This notebook validates the local project datasets and produces a small JSON report.
# It is intentionally lightweight and deterministic so it can be used as a first-run sanity check.
#
# ## Problem
# Project pipelines fail when local datasets are incomplete, mislabeled, or structurally inconsistent.
# This usually appears later during training/inference, which wastes iteration time.
#
# ## Solution
# Run deterministic checks for:
# - BUSI class availability and class balance visibility.
# - Sample-level image/mask alignment visualization.
# - Local NDT sample metadata integrity.
#
# ## Result
# The notebook produces:
# - Count/overview figures.
# - Overlay examples for quick visual verification.
# - A JSON health report that can be consumed by scripts or CI.
#
# ## How to use this notebook
# 1. Run all cells once after cloning or changing data files.
# 2. Confirm BUSI class counts are non-zero for the classes you plan to train on.
# 3. Confirm local NDT samples are detected and metadata looks plausible.
# 4. Use the generated JSON report as a machine-readable status artifact.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _notebook_utils import (
    busi_class_counts,
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_busi_sample_arrays,
    save_json_report,
    set_reproducible_seed,
    summarize_ndt_samples,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("01_dataset_healthcheck")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## BUSI class distribution

# %%
counts = busi_class_counts()
counts

# %% [markdown]
# Interpretation note:
# - Large class imbalance is expected in BUSI.
# - For classification experiments, consider weighted loss or stratified batching.
# - For segmentation experiments, always verify mask availability and quality per class.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
classes = list(counts.keys())
values = [counts[k] for k in classes]
ax.bar(classes, values, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_title("BUSI image counts (excluding masks)")
ax.set_ylabel("Number of images")
for i, v in enumerate(values):
    ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(output_dir / "busi_class_distribution.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Sample image/mask pairs

# %%
sample_classes = ["benign", "malignant"]
fig, axes = plt.subplots(len(sample_classes), 2, figsize=(8, 6))

for row, class_name in enumerate(sample_classes):
    image, mask = load_busi_sample_arrays(class_name=class_name)
    overlay = image.astype(np.float32).copy() / 255.0
    mask_bool = mask > 0
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] *= 0.3
    overlay[mask_bool, 2] *= 0.3

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{class_name}: image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title(f"{class_name}: mask overlay")
    axes[row, 1].axis("off")

plt.tight_layout()
fig.savefig(output_dir / "busi_sample_overlays.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## NDT local sample summary

# %%
ndt_rows = summarize_ndt_samples()
ndt_rows

# %% [markdown]
# Interpretation note:
# - `n_defects` is the number of known/annotated defects in each sample metadata.
# - A value of `0` does not always mean a physically perfect sample; it means no defect annotation
#   is present in the packaged metadata.

# %%
if ndt_rows:
    names = [row["sample"].replace(".npz", "") for row in ndt_rows]
    defect_counts = [row["n_defects"] for row in ndt_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, defect_counts, color="#72B7B2")
    ax.set_title("NDT samples: number of defect annotations")
    ax.set_ylabel("n_defects")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(output_dir / "ndt_defect_counts.png", dpi=140)
    plt.close(fig)

# %% [markdown]
# ## Persist healthcheck report

# %%
report = {
    "seed": seed,
    "busi_counts": counts,
    "busi_total": int(sum(counts.values())),
    "ndt_samples": ndt_rows,
}
save_json_report(output_dir / "dataset_healthcheck_report.json", report)
report

# %%
health_status = {
    "busi_ready": counts["benign"] > 0 and counts["malignant"] > 0,
    "ndt_ready": len(ndt_rows) > 0,
}
health_status["overall_ready"] = health_status["busi_ready"] and health_status["ndt_ready"]
health_status

# %% [markdown]
# ## Decision criteria
# A practical "healthy enough to proceed" state is:
# - BUSI benign and malignant counts are both non-zero.
# - At least one NDT sample is present.
# - No file-loading exceptions occurred while building overlays and summaries.
#
# ## Result interpretation
# - If `overall_ready` is `True`, the local data foundation is acceptable for notebook/model workflows.
# - If `overall_ready` is `False`, fix dataset structure first and rerun this notebook.
