"""XML-derived masks and simple segmentation baseline metrics for industrial images."""

from __future__ import annotations

from collections.abc import Iterable
from xml.etree import ElementTree

import numpy as np

from ultrasound.api.models.domain import IndustrialTrainingSampleRecord


class IndustrialSegmentationBaseline:
    """Build masks from XML annotations and score a lightweight segmentation proxy."""

    @staticmethod
    def _safe_int(value: str | None, default: int = 0) -> int:
        if value is None:
            return int(default)
        try:
            return int(round(float(value.strip())))
        except Exception:
            return int(default)

    def mask_from_annotation(
        self,
        annotation_blob: bytes,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, int]:
        height, width = int(shape[0]), int(shape[1])
        mask = np.zeros((height, width), dtype=np.uint8)
        try:
            root = ElementTree.fromstring(annotation_blob.decode("utf-8", errors="ignore"))
        except Exception:
            return mask, 0

        bbox_count = 0
        for bndbox in root.findall(".//bndbox"):
            xmin = self._safe_int(bndbox.findtext("xmin"), 0)
            ymin = self._safe_int(bndbox.findtext("ymin"), 0)
            xmax = self._safe_int(bndbox.findtext("xmax"), width - 1)
            ymax = self._safe_int(bndbox.findtext("ymax"), height - 1)

            x1 = max(0, min(width - 1, xmin))
            y1 = max(0, min(height - 1, ymin))
            x2 = max(0, min(width, xmax))
            y2 = max(0, min(height, ymax))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = 255
            bbox_count += 1

        return mask, bbox_count

    @staticmethod
    def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
        pred_bool = np.asarray(pred > 0, dtype=bool)
        target_bool = np.asarray(target > 0, dtype=bool)
        intersection = float(np.logical_and(pred_bool, target_bool).sum())
        union = float(np.logical_or(pred_bool, target_bool).sum())
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
        pred_bool = np.asarray(pred > 0, dtype=bool)
        target_bool = np.asarray(target > 0, dtype=bool)
        intersection = float(np.logical_and(pred_bool, target_bool).sum())
        denom = float(pred_bool.sum() + target_bool.sum())
        if denom <= 0.0:
            return 0.0
        return (2.0 * intersection) / denom

    def baseline_metrics(
        self,
        samples: Iterable[IndustrialTrainingSampleRecord],
        max_samples: int = 240,
    ) -> tuple[float | None, float | None, int]:
        iou_scores: list[float] = []
        dice_scores: list[float] = []
        annotated_samples = 0

        for sample_index, sample in enumerate(samples):
            if sample_index >= int(max_samples):
                break
            if sample.annotation_blob is None:
                continue

            gt_mask, bbox_count = self.mask_from_annotation(
                sample.annotation_blob,
                shape=(int(sample.image_rgb.shape[0]), int(sample.image_rgb.shape[1])),
            )
            if bbox_count <= 0:
                continue

            annotated_samples += 1
            gray = np.asarray(sample.image_rgb, dtype=np.float32).mean(axis=2)
            low_q = float(np.quantile(gray, 0.2))
            high_q = float(np.quantile(gray, 0.8))

            pred_dark = np.asarray(gray <= low_q, dtype=np.uint8) * 255
            pred_bright = np.asarray(gray >= high_q, dtype=np.uint8) * 255
            iou_dark = self.binary_iou(pred_dark, gt_mask)
            iou_bright = self.binary_iou(pred_bright, gt_mask)
            pred_mask = pred_dark if iou_dark >= iou_bright else pred_bright

            iou_scores.append(self.binary_iou(pred_mask, gt_mask))
            dice_scores.append(self.binary_dice(pred_mask, gt_mask))

        if not iou_scores:
            return None, None, annotated_samples

        return float(np.mean(iou_scores)), float(np.mean(dice_scores)), annotated_samples
