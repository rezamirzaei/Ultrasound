"""Reusable mini training smoke workflow for BUSI segmentation/classification paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ultrasound.models.classifier import UltrasoundClassifier
from ultrasound.models.unet import UNetSmall, combined_loss


@dataclass(frozen=True)
class MiniTrainingPipelineResult:
    segmentation_losses: list[float]
    classification_losses: list[float]
    report: dict[str, Any]
    status: dict[str, bool | int]


def run_mini_training_pipeline(
    loader: Any,
    *,
    seed: int = 42,
    steps: int = 2,
    learning_rate: float = 2e-4,
    device: str = "cpu",
) -> MiniTrainingPipelineResult:
    torch.manual_seed(seed)
    target_device = torch.device(device)

    seg_model = UNetSmall(in_channels=3, out_channels=1, features=[16, 32, 64, 128]).to(
        target_device
    )
    seg_opt = Adam(seg_model.parameters(), lr=learning_rate)

    seg_losses: list[float] = []
    for step, (images, masks, _) in enumerate(loader):
        if step >= int(steps):
            break
        images = images.to(target_device)
        masks = masks.to(target_device)

        seg_opt.zero_grad(set_to_none=True)
        logits = seg_model(images)
        loss = combined_loss(logits, masks, bce_weight=0.5)
        loss.backward()
        seg_opt.step()
        seg_losses.append(float(loss.item()))

    clf_model = UltrasoundClassifier(num_classes=2, in_channels=3, dropout=0.25).to(target_device)
    clf_opt = Adam(clf_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    clf_losses: list[float] = []
    for step, (images, _, labels) in enumerate(loader):
        if step >= int(steps):
            break
        images = images.to(target_device)
        labels = labels.to(target_device)

        clf_opt.zero_grad(set_to_none=True)
        logits = clf_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        clf_opt.step()
        clf_losses.append(float(loss.item()))

    status: dict[str, bool | int] = {
        "segmentation_steps": len(seg_losses),
        "classification_steps": len(clf_losses),
        "segmentation_losses_finite": bool(np.all(np.isfinite(seg_losses))) if seg_losses else False,
        "classification_losses_finite": bool(np.all(np.isfinite(clf_losses))) if clf_losses else False,
    }
    status["overall_pass"] = bool(
        int(status["segmentation_steps"]) > 0
        and int(status["classification_steps"]) > 0
        and bool(status["segmentation_losses_finite"])
        and bool(status["classification_losses_finite"])
    )

    report = {
        "seed": int(seed),
        "segmentation_loss": seg_losses,
        "classification_loss": clf_losses,
    }
    return MiniTrainingPipelineResult(
        segmentation_losses=seg_losses,
        classification_losses=clf_losses,
        report=report,
        status=status,
    )
