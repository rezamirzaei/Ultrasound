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
# # 04. Mini Training Pipeline (Smoke)
# Thin notebook wrapper around the reusable training smoke workflow.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from _notebook_utils import (
    busi_class_counts,
    ensure_notebook_output_dir,
    ensure_src_on_path,
    save_json_report,
    set_reproducible_seed,
)
from torch.utils.data import DataLoader

project_root = ensure_src_on_path()
from ultrasound.data import BUSIDataset
from ultrasound.utils.visualization import plot_training_history
from ultrasound.workflows import run_mini_training_pipeline

seed = set_reproducible_seed(42)
torch.manual_seed(seed)
output_dir = ensure_notebook_output_dir("04_mini_training_pipeline")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
counts = busi_class_counts()
if counts["benign"] == 0 or counts["malignant"] == 0:
    raise RuntimeError("BUSI data is missing. Expected data under data/busi/{benign,malignant}.")

transform, mask_transform = BUSIDataset.get_default_transforms(image_size=96, augment=False)
dataset = BUSIDataset(
    root_dir=str(project_root / "data" / "busi"),
    split="train",
    transform=transform,
    mask_transform=mask_transform,
    include_normal=False,
    binary_classification=True,
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

result = run_mini_training_pipeline(loader, seed=seed, steps=2, device="cpu")
report = dict(result.report)
report["dataset_size"] = int(len(dataset))
report

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(result.segmentation_losses, marker="o", color="#4C78A8")
axes[0].set_title("Segmentation loss")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(result.classification_losses, marker="o", color="#F58518")
axes[1].set_title("Classification loss")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "mini_training_losses.png", dpi=140)
plt.close(fig)

plot_training_history(
    {
        "loss": result.segmentation_losses,
        "val_loss": list(reversed(result.segmentation_losses)),
    },
    metrics=["loss"],
    save_path=str(output_dir / "training_history_style_plot.png"),
)

# %%
save_json_report(output_dir / "mini_training_report.json", report)
result.status
