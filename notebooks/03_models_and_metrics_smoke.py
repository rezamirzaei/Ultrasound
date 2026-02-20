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
# # 03. Models and Metrics Smoke Test
#
# Small forward-pass and metric checks for model/API correctness.
#
# ## Problem
# Model pipelines often fail due to shape mismatches, invalid loss wiring, or metric misuse.
# These failures are easy to miss until expensive training runs.
#
# ## Solution
# Execute a compact smoke suite:
# - Forward pass through segmentation and classification models.
# - Loss computation checks.
# - Basic segmentation/classification metric checks with known synthetic examples.
#
# ## Result
# A successful run confirms that core model interfaces and metric functions are operational
# before launching longer experiments.
#
# ## Scope
# This notebook verifies model and metric plumbing.
# It does not measure final model quality, and it intentionally avoids long training.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_busi_sample_arrays,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
torch.manual_seed(seed)
output_dir = ensure_notebook_output_dir("03_models_and_metrics_smoke")

from ultrasound.models.classifier import ResNetClassifier, UltrasoundClassifier, focal_loss
from ultrasound.models.unet import AttentionUNet, UNet, combined_loss, dice_loss
from ultrasound.utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_segmentation_metrics,
)
from ultrasound.utils.visualization import plot_confusion_matrix

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Forward pass validation

# %%
x = torch.randn(2, 3, 128, 128)

unet = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])
attn_unet = AttentionUNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])
cnn_classifier = UltrasoundClassifier(num_classes=2, in_channels=3, dropout=0.3)
resnet_classifier = ResNetClassifier(
    num_classes=2,
    pretrained=False,
    model_name="resnet18",
    freeze_backbone=False,
    dropout=0.3,
)

with torch.no_grad():
    unet_logits = unet(x)
    attn_logits = attn_unet(x)
    cnn_logits = cnn_classifier(x)
    resnet_logits = resnet_classifier(x)

shapes = {
    "input": list(x.shape),
    "unet_logits": list(unet_logits.shape),
    "attention_unet_logits": list(attn_logits.shape),
    "cnn_logits": list(cnn_logits.shape),
    "resnet_logits": list(resnet_logits.shape),
}
shapes

# %% [markdown]
# Interpretation note:
# - Segmentation logits should be shape `[N, 1, H, W]`.
# - Classification logits should be shape `[N, C]`.
# - Any mismatch here usually indicates an API or architecture wiring error.

# %% [markdown]
# ## Loss sanity check

# %%
seg_target = torch.randint(0, 2, (2, 1, 128, 128)).float()
class_target = torch.randint(0, 2, (2,), dtype=torch.long)

losses = {
    "dice_loss": float(dice_loss(unet_logits, seg_target).item()),
    "combined_loss": float(combined_loss(unet_logits, seg_target, bce_weight=0.5).item()),
    "focal_loss": float(focal_loss(cnn_logits, class_target, alpha=0.25, gamma=2.0).item()),
}
losses

# %% [markdown]
# Interpretation note:
# - Loss values only need to be finite and positive in this smoke test.
# - Stability and monotonic improvement are evaluated in longer training experiments.

# %% [markdown]
# ## Metric sanity check with one BUSI mask

# %%
image, mask = load_busi_sample_arrays(class_name="benign")
mask_bin = (mask > 0).astype(np.uint8)

# Synthetic prediction for smoke check: slight horizontal shift.
pred_mask = np.roll(mask_bin, shift=2, axis=1)

seg_metrics = compute_segmentation_metrics(pred_mask, mask_bin)
seg_metrics

# %% [markdown]
# ## Classification metrics smoke check

# %%
y_true = np.array([0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1, 0, 1])

class_metrics = compute_classification_metrics(y_pred, y_true, class_names=["benign", "malignant"])
cm = compute_confusion_matrix(y_pred, y_true)

fig = plot_confusion_matrix(cm, class_names=["benign", "malignant"], normalize=False)
fig.savefig(output_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight")
plt.close(fig)

class_metrics

# %% [markdown]
# Practical reading:
# - Use confusion matrix counts to confirm class mapping consistency.
# - Use macro metrics to detect imbalanced performance between benign and malignant classes.

# %% [markdown]
# ## Persist smoke report

# %%
report = {
    "seed": seed,
    "shapes": shapes,
    "losses": losses,
    "segmentation_metrics": seg_metrics,
    "classification_metrics": class_metrics,
    "confusion_matrix": cm.tolist(),
}

save_json_report(output_dir / "model_metric_smoke_report.json", report)
report

# %%
smoke_status = {
    "shape_checks_passed": (
        shapes["unet_logits"][1] == 1
        and len(shapes["unet_logits"]) == 4
        and len(shapes["resnet_logits"]) == 2
        and shapes["resnet_logits"][1] == 2
    ),
    "losses_finite": all(np.isfinite(list(losses.values()))),
    "metric_values_finite": all(
        np.isfinite(
            [
                seg_metrics["dice"],
                seg_metrics["iou"],
                seg_metrics["pixel_accuracy"],
                class_metrics["accuracy"],
                class_metrics["macro_f1"],
            ]
        )
    ),
}
smoke_status["overall_pass"] = (
    smoke_status["shape_checks_passed"]
    and smoke_status["losses_finite"]
    and smoke_status["metric_values_finite"]
)
smoke_status

# %% [markdown]
# ## Result interpretation
# - If `overall_pass` is `True`, model/metric plumbing is ready for longer training runs.
# - If `overall_pass` is `False`, fix interface or tensor-shape issues before scaling up.
