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
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    load_busi_sample_arrays,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("03_models_and_metrics_smoke")

from ultrasound.utils.visualization import plot_confusion_matrix
from ultrasound.workflows import run_model_metric_smoke

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Forward pass validation

# %%
image, mask = load_busi_sample_arrays(class_name="benign")
smoke = run_model_metric_smoke(image, mask, seed=seed)
shapes = smoke.shapes
shapes

# %% [markdown]
# Interpretation note:
# - Segmentation logits should be shape `[N, 1, H, W]`.
# - Classification logits should be shape `[N, C]`.
# - Any mismatch here usually indicates an API or architecture wiring error.

# %% [markdown]
# ## Loss sanity check

# %%
losses = smoke.losses
losses

# %% [markdown]
# Interpretation note:
# - Loss values only need to be finite and positive in this smoke test.
# - Stability and monotonic improvement are evaluated in longer training experiments.

# %% [markdown]
# ## Metric sanity check with one BUSI mask

# %%
seg_metrics = smoke.segmentation_metrics
seg_metrics

# %% [markdown]
# ## Classification metrics smoke check

# %%
class_metrics = smoke.classification_metrics
cm = smoke.confusion_matrix

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
