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
#
# Minimal training loops to verify data loading, loss computation, and optimization steps.
# This notebook is intentionally short and deterministic.
#
# ## Problem
# Full training jobs are expensive; running them without a pipeline sanity check can waste hours.
# We need a fast way to verify that dataloaders, models, losses, and optimizers interact correctly.
#
# ## Solution
# Run a very small deterministic training loop (2 steps) for:
# - Segmentation model path.
# - Classification model path.
#
# ## Result
# The notebook outputs short loss traces and a report indicating whether the training path executes
# without NaN/Inf issues or runtime failures.
#
# ## Why this exists
# Full training notebooks are expensive and sensitive to environment differences.
# This smoke version checks end-to-end training mechanics quickly on CPU.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from _notebook_utils import (
    busi_class_counts,
    ensure_notebook_output_dir,
    ensure_src_on_path,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
torch.manual_seed(seed)
output_dir = ensure_notebook_output_dir("04_mini_training_pipeline")

from ultrasound.data import BUSIDataset
from ultrasound.models.classifier import UltrasoundClassifier
from ultrasound.models.unet import UNetSmall, combined_loss
from ultrasound.utils.visualization import plot_training_history

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Build lightweight dataloader

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

print(f"Training samples: {len(dataset)}")

# %% [markdown]
# Interpretation note:
# - This notebook intentionally uses tiny batch count and only two optimization steps.
# - The goal is to verify the training path is executable, not to achieve convergence.

# %% [markdown]
# ## Segmentation smoke training (2 steps)

# %%
device = torch.device("cpu")
seg_model = UNetSmall(in_channels=3, out_channels=1, features=[16, 32, 64, 128]).to(device)
seg_opt = Adam(seg_model.parameters(), lr=2e-4)

seg_losses: list[float] = []
for step, (images, masks, _) in enumerate(loader):
    if step >= 2:
        break
    images = images.to(device)
    masks = masks.to(device)

    seg_opt.zero_grad(set_to_none=True)
    logits = seg_model(images)
    loss = combined_loss(logits, masks, bce_weight=0.5)
    loss.backward()
    seg_opt.step()

    seg_losses.append(float(loss.item()))

seg_losses

# %% [markdown]
# Sanity check:
# - Values should be finite (`not NaN`, `not inf`).
# - A small step-to-step fluctuation is normal with tiny batches.

# %% [markdown]
# ## Classification smoke training (2 steps)

# %%
clf_model = UltrasoundClassifier(num_classes=2, in_channels=3, dropout=0.25).to(device)
clf_opt = Adam(clf_model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

clf_losses: list[float] = []
for step, (images, _, labels) in enumerate(loader):
    if step >= 2:
        break
    images = images.to(device)
    labels = labels.to(device)

    clf_opt.zero_grad(set_to_none=True)
    logits = clf_model(images)
    loss = criterion(logits, labels)
    loss.backward()
    clf_opt.step()

    clf_losses.append(float(loss.item()))

clf_losses

# %% [markdown]
# Sanity check:
# - Values should remain finite.
# - Do not interpret these two-step losses as model performance indicators.

# %% [markdown]
# ## Plot and save run summary

# %%
history = {
    "segmentation_loss": seg_losses,
    "classification_loss": clf_losses,
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(seg_losses, marker="o", color="#4C78A8")
axes[0].set_title("Segmentation loss")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(clf_losses, marker="o", color="#F58518")
axes[1].set_title("Classification loss")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "mini_training_losses.png", dpi=140)
plt.close(fig)

plot_training_history(
    {
        "loss": seg_losses,
        "val_loss": list(reversed(seg_losses)),
    },
    metrics=["loss"],
    save_path=str(output_dir / "training_history_style_plot.png"),
)

report = {
    "seed": seed,
    "dataset_size": int(len(dataset)),
    "segmentation_loss": seg_losses,
    "classification_loss": clf_losses,
}
save_json_report(output_dir / "mini_training_report.json", report)
report

# %%
train_status = {
    "segmentation_steps": len(seg_losses),
    "classification_steps": len(clf_losses),
    "segmentation_losses_finite": bool(np.all(np.isfinite(seg_losses))) if seg_losses else False,
    "classification_losses_finite": bool(np.all(np.isfinite(clf_losses))) if clf_losses else False,
}
train_status["overall_pass"] = (
    train_status["segmentation_steps"] > 0
    and train_status["classification_steps"] > 0
    and train_status["segmentation_losses_finite"]
    and train_status["classification_losses_finite"]
)
train_status

# %% [markdown]
# ## Next step
# Run a longer, stratified training job (with validation split and checkpointing)
# once this smoke notebook passes.
#
# ## Result interpretation
# - If `overall_pass` is `True`, the core training loop is healthy.
# - If `overall_pass` is `False`, debug data/model/loss plumbing before any long run.
