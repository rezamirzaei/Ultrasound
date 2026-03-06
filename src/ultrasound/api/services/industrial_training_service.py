"""Industrial dataset learning service (classification + annotation-driven segmentation baseline)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal, cast
from xml.etree import ElementTree

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

from ultrasound.api.models.domain import (
    IndustrialTrainingCurvePointRecord,
    IndustrialTrainingRunRecord,
    IndustrialTrainingSampleRecord,
)
from ultrasound.api.models.schemas import (
    IndustrialSegmentationPreview,
    IndustrialTrainingCurvePoint,
    IndustrialTrainingRequest,
    IndustrialTrainingResponse,
)
from ultrasound.api.services.interfaces import IndustrialTrainingRepository, MediaRenderer


class IndustrialTrainingService:
    """Runs lightweight industrial learning and exposes segmentation diagnostics."""

    def __init__(self, dataset_repository: IndustrialTrainingRepository, media_service: MediaRenderer):
        self.dataset_repository = dataset_repository
        self.media_service = media_service

    def _resolve_task_profile(
        self,
        dataset_name: str,
        class_labels: list[str],
        annotated_samples: int,
    ) -> tuple[
        Literal["classification_single_label", "classification_single_label_with_bbox"],
        Literal["binary", "multiclass"],
        Literal["folder_name", "folder_name_plus_xml_bbox"],
        bool,
        str,
    ]:
        segmentation_supported = int(annotated_samples) > 0
        task_type: Literal["classification_single_label", "classification_single_label_with_bbox"]
        label_source: Literal["folder_name", "folder_name_plus_xml_bbox"]
        if segmentation_supported:
            task_type = "classification_single_label_with_bbox"
            label_source = "folder_name_plus_xml_bbox"
        else:
            task_type = "classification_single_label"
            label_source = "folder_name"

        classification_mode: Literal["binary", "multiclass"] = (
            "binary" if len(class_labels) == 2 else "multiclass"
        )

        if segmentation_supported:
            segmentation_notes = (
                "Annotation labels are available: segmentation masks are derived from XML "
                "bounding boxes."
            )
        elif dataset_name == "casting_defect":
            segmentation_notes = (
                "Casting dataset is classification-only in current storage "
                "(def_front vs ok_front); no segmentation labels found."
            )
        elif dataset_name == "steel_defect":
            segmentation_notes = (
                "Steel dataset labels come from train/valid/test class folders; "
                "Thumbs.db is ignored and does not provide supervised labels."
            )
        else:
            segmentation_notes = (
                "No annotation labels were found in current SQL storage; "
                "segmentation is unavailable for this dataset."
            )

        return (
            task_type,
            classification_mode,
            label_source,
            segmentation_supported,
            segmentation_notes,
        )

    def _extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = np.asarray(image_rgb, dtype=np.float32).mean(axis=2)
        gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)

        small = Image.fromarray(gray, mode="L").resize((48, 48), Image.Resampling.BILINEAR)
        small_arr = np.asarray(small, dtype=np.float32) / 255.0

        grad_x = np.diff(small_arr, axis=1, append=small_arr[:, -1:])
        grad_y = np.diff(small_arr, axis=0, append=small_arr[-1:, :])
        grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))

        percentile_features = np.percentile(small_arr, [10, 25, 50, 75, 90]).astype(np.float32)
        stats = np.array(
            [
                float(np.mean(small_arr)),
                float(np.std(small_arr)),
                float(np.mean(grad_mag)),
                float(np.std(grad_mag)),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [small_arr.reshape(-1), grad_mag.reshape(-1), percentile_features, stats], axis=0
        )

    def _prepare_dataset(self, dataset_name: str) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, int],
        list[str],
        list[IndustrialTrainingSampleRecord],
        list[IndustrialTrainingSampleRecord],
    ]:
        samples, class_counts, class_labels = (
            self.dataset_repository.list_industrial_training_samples(dataset_name)
        )
        train_samples = [sample for sample in samples if sample.split == "train"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        if not train_samples or not test_samples:
            raise ValueError(
                f"Dataset '{dataset_name}' must provide both train and test samples for learning."
            )

        train_labels = sorted({sample.label for sample in train_samples})
        test_labels = sorted({sample.label for sample in test_samples})
        if len(train_labels) < 2:
            raise ValueError(
                f"Dataset '{dataset_name}' needs at least two classes in train split for learning."
            )
        if not set(test_labels).issubset(set(train_labels)):
            raise ValueError(
                "Industrial train split does not cover all classes present in test split."
            )

        x_train = np.vstack([self._extract_features(sample.image_rgb) for sample in train_samples])
        y_train = np.asarray([sample.label for sample in train_samples], dtype=np.int64)
        x_test = np.vstack([self._extract_features(sample.image_rgb) for sample in test_samples])
        y_test = np.asarray([sample.label for sample in test_samples], dtype=np.int64)

        mean = np.mean(x_train, axis=0, keepdims=True)
        std = np.std(x_train, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        return (
            x_train,
            y_train,
            x_test,
            y_test,
            class_counts,
            class_labels,
            train_samples,
            test_samples,
        )

    def _evaluate_epoch(
        self, model: SGDClassifier, features: np.ndarray, labels: np.ndarray, all_labels: np.ndarray
    ) -> tuple[float, float]:
        pred = model.predict(features)
        acc = float(accuracy_score(labels, pred))
        probabilities = model.predict_proba(features)
        loss = float(log_loss(labels, probabilities, labels=list(all_labels)))
        return acc, loss

    def _run_training(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> list[IndustrialTrainingCurvePointRecord]:
        rng = np.random.default_rng(42)
        labels = np.asarray(sorted(np.unique(y_train)), dtype=np.int64)

        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1,
            learning_rate="constant",
            eta0=float(learning_rate),
            random_state=42,
            fit_intercept=True,
            warm_start=True,
            tol=None,
        )

        curve: list[IndustrialTrainingCurvePointRecord] = []
        initialized = False

        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, x_train.shape[0], int(batch_size)):
                batch_idx = order[start : start + int(batch_size)]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                if not initialized:
                    model.partial_fit(x_batch, y_batch, classes=labels)
                    initialized = True
                else:
                    model.partial_fit(x_batch, y_batch)

            train_acc, train_loss = self._evaluate_epoch(model, x_train, y_train, labels)
            test_acc, test_loss = self._evaluate_epoch(model, x_test, y_test, labels)
            curve.append(
                IndustrialTrainingCurvePointRecord(
                    epoch=epoch,
                    train_accuracy=train_acc,
                    test_accuracy=test_acc,
                    train_loss=train_loss,
                    test_loss=test_loss,
                )
            )

        return curve

    def _safe_int(self, value: str | None, default: int = 0) -> int:
        if value is None:
            return int(default)
        try:
            return int(round(float(value.strip())))
        except Exception:
            return int(default)

    def _mask_from_annotation(
        self, annotation_blob: bytes, shape: tuple[int, int]
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

    def _binary_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        pred_bool = np.asarray(pred > 0, dtype=bool)
        target_bool = np.asarray(target > 0, dtype=bool)
        intersection = float(np.logical_and(pred_bool, target_bool).sum())
        union = float(np.logical_or(pred_bool, target_bool).sum())
        if union <= 0.0:
            return 0.0
        return intersection / union

    def _binary_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        pred_bool = np.asarray(pred > 0, dtype=bool)
        target_bool = np.asarray(target > 0, dtype=bool)
        intersection = float(np.logical_and(pred_bool, target_bool).sum())
        denom = float(pred_bool.sum() + target_bool.sum())
        if denom <= 0.0:
            return 0.0
        return (2.0 * intersection) / denom

    def _baseline_segmentation_metrics(
        self, samples: Iterable[IndustrialTrainingSampleRecord], max_samples: int = 240
    ) -> tuple[float | None, float | None, int]:
        iou_scores: list[float] = []
        dice_scores: list[float] = []
        annotated_samples = 0

        for sample_index, sample in enumerate(samples):
            if sample_index >= int(max_samples):
                break
            if sample.annotation_blob is None:
                continue
            gt_mask, bbox_count = self._mask_from_annotation(
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
            iou_dark = self._binary_iou(pred_dark, gt_mask)
            iou_bright = self._binary_iou(pred_bright, gt_mask)
            pred_mask = pred_dark if iou_dark >= iou_bright else pred_bright

            iou_scores.append(self._binary_iou(pred_mask, gt_mask))
            dice_scores.append(self._binary_dice(pred_mask, gt_mask))

        if not iou_scores:
            return None, None, annotated_samples

        return float(np.mean(iou_scores)), float(np.mean(dice_scores)), annotated_samples

    def _to_response(self, run: IndustrialTrainingRunRecord) -> IndustrialTrainingResponse:
        return IndustrialTrainingResponse(
            run_id=run.run_id,
            generated_at=run.created_at,
            storage="sql",
            dataset_name=run.dataset_name,
            epochs=run.epochs,
            batch_size=run.batch_size,
            learning_rate=run.learning_rate,
            train_samples=run.train_samples,
            test_samples=run.test_samples,
            class_counts=run.class_counts,
            class_labels=run.class_labels,
            task_type=run.task_type,
            classification_mode=run.classification_mode,
            label_source=run.label_source,
            segmentation_supported=run.segmentation_supported,
            segmentation_notes=run.segmentation_notes,
            train_accuracy=run.train_accuracy,
            test_accuracy=run.test_accuracy,
            train_loss=run.train_loss,
            test_loss=run.test_loss,
            curve=[
                IndustrialTrainingCurvePoint(
                    epoch=point.epoch,
                    train_accuracy=point.train_accuracy,
                    test_accuracy=point.test_accuracy,
                    train_loss=point.train_loss,
                    test_loss=point.test_loss,
                )
                for point in run.curve
            ],
            annotated_samples=run.annotated_samples,
            segmentation_iou_train=run.segmentation_iou_train,
            segmentation_iou_test=run.segmentation_iou_test,
            segmentation_dice_train=run.segmentation_dice_train,
            segmentation_dice_test=run.segmentation_dice_test,
            notes=run.notes,
        )

    def get_latest_run(self, dataset_name: str) -> IndustrialTrainingResponse:
        latest = self.dataset_repository.get_latest_industrial_training_run(dataset_name)
        if latest is not None:
            return self._to_response(latest)

        counts = self.dataset_repository.get_industrial_counts().get(dataset_name, {})
        class_counts: dict[str, int] = {}
        for class_map in counts.values():
            for class_name, n in class_map.items():
                class_counts[class_name] = int(class_counts.get(class_name, 0) + int(n))
        class_labels = sorted(class_counts.keys())
        annotation_count = self.dataset_repository.get_industrial_annotation_count(dataset_name)
        (
            task_type,
            classification_mode,
            label_source,
            segmentation_supported,
            segmentation_notes,
        ) = self._resolve_task_profile(
            dataset_name=dataset_name,
            class_labels=class_labels,
            annotated_samples=annotation_count,
        )

        return IndustrialTrainingResponse(
            run_id=None,
            generated_at=datetime.now(timezone.utc),
            storage="sql",
            dataset_name=cast(
                Literal["steel_defect", "neu_surface", "casting_defect"],
                dataset_name,
            ),
            epochs=0,
            batch_size=0,
            learning_rate=0.0,
            train_samples=0,
            test_samples=0,
            class_counts=class_counts,
            class_labels=class_labels,
            task_type=task_type,
            classification_mode=classification_mode,
            label_source=label_source,
            segmentation_supported=segmentation_supported,
            segmentation_notes=segmentation_notes,
            train_accuracy=None,
            test_accuracy=None,
            train_loss=None,
            test_loss=None,
            curve=[],
            annotated_samples=int(annotation_count),
            segmentation_iou_train=None,
            segmentation_iou_test=None,
            segmentation_dice_train=None,
            segmentation_dice_test=None,
            notes="No industrial learning run found. Queue /learning/jobs/industrial-training first.",
        )

    def run_training(self, request: IndustrialTrainingRequest) -> IndustrialTrainingResponse:
        (
            x_train,
            y_train,
            x_test,
            y_test,
            class_counts,
            class_labels,
            train_samples,
            test_samples,
        ) = self._prepare_dataset(request.dataset_name)

        curve = self._run_training(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        )
        if not curve:
            raise ValueError("Industrial training produced no curve points.")

        has_annotation_blob = any(
            sample.annotation_blob is not None for sample in train_samples
        ) or any(sample.annotation_blob is not None for sample in test_samples)
        if has_annotation_blob:
            seg_iou_train, seg_dice_train, ann_train = self._baseline_segmentation_metrics(
                train_samples
            )
            seg_iou_test, seg_dice_test, ann_test = self._baseline_segmentation_metrics(
                test_samples
            )
        else:
            seg_iou_train, seg_dice_train, ann_train = None, None, 0
            seg_iou_test, seg_dice_test, ann_test = None, None, 0
        annotated_samples = int(ann_train + ann_test)
        (
            task_type,
            classification_mode,
            label_source,
            segmentation_supported,
            segmentation_notes,
        ) = self._resolve_task_profile(
            dataset_name=request.dataset_name,
            class_labels=class_labels,
            annotated_samples=annotated_samples,
        )

        notes = (
            "Classification is SGD(log-loss) over deterministic SQL splits. " + segmentation_notes
        )
        last = curve[-1]
        run = IndustrialTrainingRunRecord(
            created_at=datetime.now(timezone.utc),
            dataset_name=request.dataset_name,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            train_samples=int(x_train.shape[0]),
            test_samples=int(x_test.shape[0]),
            class_counts=class_counts,
            class_labels=class_labels,
            task_type=task_type,
            classification_mode=classification_mode,
            label_source=label_source,
            segmentation_supported=segmentation_supported,
            segmentation_notes=segmentation_notes,
            train_accuracy=last.train_accuracy,
            test_accuracy=last.test_accuracy,
            train_loss=last.train_loss,
            test_loss=last.test_loss,
            curve=curve,
            annotated_samples=annotated_samples,
            segmentation_iou_train=seg_iou_train,
            segmentation_iou_test=seg_iou_test,
            segmentation_dice_train=seg_dice_train,
            segmentation_dice_test=seg_dice_test,
            notes=notes,
        )

        persisted = self.dataset_repository.save_industrial_training_run(run)
        return self._to_response(persisted)

    def get_segmentation_preview(
        self,
        dataset_name: str,
        split: str,
        class_name: str,
        sample_index: int,
    ) -> IndustrialSegmentationPreview:
        sample = self.dataset_repository.get_industrial_sample(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            index=sample_index,
        )
        annotation_count = self.dataset_repository.get_industrial_annotation_count(dataset_name)
        class_counts = self.dataset_repository.get_industrial_counts().get(dataset_name, {})
        class_labels = sorted(
            {class_name_key for split_map in class_counts.values() for class_name_key in split_map}
        )
        task_type, _, _, segmentation_supported, segmentation_notes = self._resolve_task_profile(
            dataset_name=dataset_name,
            class_labels=class_labels,
            annotated_samples=annotation_count,
        )

        mask = np.zeros(sample.image_rgb.shape[:2], dtype=np.uint8)
        bbox_count = 0
        source: Literal["annotation_xml", "none"] = "none"

        if sample.annotation_blob:
            parsed_mask, parsed_bbox_count = self._mask_from_annotation(
                sample.annotation_blob,
                shape=(int(sample.image_rgb.shape[0]), int(sample.image_rgb.shape[1])),
            )
            if parsed_bbox_count > 0:
                mask = parsed_mask
                bbox_count = int(parsed_bbox_count)
                source = "annotation_xml"

        coverage = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
        if not segmentation_supported:
            message = segmentation_notes
        elif source == "none":
            message = (
                "Segmentation is supported for this dataset, but this sample has no valid "
                "annotation bbox."
            )
        else:
            message = "Segmentation mask rendered from annotation XML bounding boxes."

        return IndustrialSegmentationPreview(
            dataset_name=sample.dataset_name,
            split=sample.split,
            class_name=sample.class_name,
            requested_index=sample.requested_index,
            resolved_index=sample.resolved_index,
            total_samples=sample.total_samples,
            image_shape=[int(v) for v in sample.image_rgb.shape],
            bbox_count=bbox_count,
            annotation_coverage_ratio=coverage,
            task_type=task_type,
            segmentation_supported=segmentation_supported,
            message=message,
            source=source,
            image_data_url=self.media_service.as_png_data_url(sample.image_rgb),
            mask_data_url=self.media_service.as_png_data_url(mask),
            relative_path=sample.relative_path,
        )
