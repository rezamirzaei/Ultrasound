"""Reusable smoke workflow for model wiring and metric validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ultrasound.models.classifier import ResNetClassifier, UltrasoundClassifier, focal_loss
from ultrasound.models.unet import AttentionUNet, UNet, combined_loss, dice_loss
from ultrasound.utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_segmentation_metrics,
)


@dataclass(frozen=True)
class ModelMetricSmokeResult:
    shapes: dict[str, list[int]]
    losses: dict[str, float]
    segmentation_metrics: dict[str, float]
    classification_metrics: dict[str, float]
    confusion_matrix: np.ndarray


def run_model_metric_smoke(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int = 42,
) -> ModelMetricSmokeResult:
    image = np.asarray(image_rgb, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape [H, W, 3]")
    if mask_arr.ndim != 2:
        raise ValueError("mask must be single-channel")

    torch.manual_seed(seed)
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

    seg_target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    class_target = torch.randint(0, 2, (2,), dtype=torch.long)
    losses = {
        "dice_loss": float(dice_loss(unet_logits, seg_target).item()),
        "combined_loss": float(combined_loss(unet_logits, seg_target, bce_weight=0.5).item()),
        "focal_loss": float(focal_loss(cnn_logits, class_target, alpha=0.25, gamma=2.0).item()),
    }

    mask_bin = (mask_arr > 0).astype(np.uint8)
    pred_mask = np.roll(mask_bin, shift=2, axis=1)
    seg_metrics = {
        key: float(value)
        for key, value in compute_segmentation_metrics(pred_mask, mask_bin).items()
    }

    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1])
    class_metrics = {
        key: float(value)
        for key, value in compute_classification_metrics(
            y_pred,
            y_true,
            class_names=["benign", "malignant"],
        ).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    confusion_matrix = compute_confusion_matrix(y_pred, y_true)

    return ModelMetricSmokeResult(
        shapes=shapes,
        losses=losses,
        segmentation_metrics=seg_metrics,
        classification_metrics=class_metrics,
        confusion_matrix=confusion_matrix,
    )
