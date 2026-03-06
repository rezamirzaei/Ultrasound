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
# Thin notebook wrapper around the reusable dataset healthcheck workflow.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
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
from ultrasound.workflows import run_dataset_healthcheck

seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("01_dataset_healthcheck")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
counts = busi_class_counts()
ndt_rows = summarize_ndt_samples()
sample_pairs = {
    class_name: load_busi_sample_arrays(class_name=class_name)
    for class_name in ("benign", "malignant")
}
result = run_dataset_healthcheck(counts, sample_pairs, ndt_rows, seed=seed)
result.report

# %%
fig, ax = plt.subplots(figsize=(6, 4))
classes = list(result.busi_counts.keys())
values = [result.busi_counts[name] for name in classes]
ax.bar(classes, values, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_title("BUSI image counts (excluding masks)")
ax.set_ylabel("Number of images")
for index, value in enumerate(values):
    ax.text(index, value + 2, str(value), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(output_dir / "busi_class_distribution.png", dpi=140)
plt.close(fig)

# %%
fig, axes = plt.subplots(len(sample_pairs), 2, figsize=(8, 6))
for row, (class_name, (image, _mask)) in enumerate(sample_pairs.items()):
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{class_name}: image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(result.overlays[class_name])
    axes[row, 1].set_title(f"{class_name}: mask overlay")
    axes[row, 1].axis("off")
plt.tight_layout()
fig.savefig(output_dir / "busi_sample_overlays.png", dpi=140)
plt.close(fig)

# %%
if result.ndt_rows:
    names = [row["sample"].replace(".npz", "") for row in result.ndt_rows]
    defect_counts = [row["n_defects"] for row in result.ndt_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, defect_counts, color="#72B7B2")
    ax.set_title("NDT samples: number of defect annotations")
    ax.set_ylabel("n_defects")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(output_dir / "ndt_defect_counts.png", dpi=140)
    plt.close(fig)

# %%
save_json_report(output_dir / "dataset_healthcheck_report.json", result.report)
result.health_status
