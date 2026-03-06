"""Industrial dataset learning service (classification + annotation-driven segmentation baseline)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, cast

import numpy as np
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
from ultrasound.api.services.industrial_feature_extractor import IndustrialFeatureExtractor
from ultrasound.api.services.industrial_segmentation_baseline import IndustrialSegmentationBaseline
from ultrasound.api.services.industrial_task_profile import resolve_industrial_task_profile
from ultrasound.api.services.interfaces import IndustrialTrainingRepository, MediaRenderer
from ultrasound.api.services.service_errors import InvalidRequestError, NotFoundError


class IndustrialTrainingService:
    """Runs lightweight industrial learning and exposes segmentation diagnostics."""

    def __init__(
        self,
        dataset_repository: IndustrialTrainingRepository,
        media_service: MediaRenderer,
        feature_extractor: IndustrialFeatureExtractor | None = None,
        segmentation_baseline: IndustrialSegmentationBaseline | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.media_service = media_service
        self.feature_extractor = feature_extractor or IndustrialFeatureExtractor()
        self.segmentation_baseline = segmentation_baseline or IndustrialSegmentationBaseline()

    def _map_repository_error(self, exc: Exception) -> Exception:
        if isinstance(exc, FileNotFoundError):
            return NotFoundError(str(exc))
        if isinstance(exc, ValueError):
            return InvalidRequestError(str(exc))
        return exc

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
            raise InvalidRequestError(
                f"Dataset '{dataset_name}' must provide both train and test samples for learning."
            )

        train_labels = sorted({sample.label for sample in train_samples})
        test_labels = sorted({sample.label for sample in test_samples})
        if len(train_labels) < 2:
            raise InvalidRequestError(
                f"Dataset '{dataset_name}' needs at least two classes in train split for learning."
            )
        if not set(test_labels).issubset(set(train_labels)):
            raise InvalidRequestError(
                "Industrial train split does not cover all classes present in test split."
            )

        x_train = np.vstack(
            [self.feature_extractor.extract(sample.image_rgb) for sample in train_samples]
        )
        y_train = np.asarray([sample.label for sample in train_samples], dtype=np.int64)
        x_test = np.vstack(
            [self.feature_extractor.extract(sample.image_rgb) for sample in test_samples]
        )
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
        try:
            latest = self.dataset_repository.get_latest_industrial_training_run(dataset_name)
            counts = self.dataset_repository.get_industrial_counts().get(dataset_name, {})
            annotation_count = self.dataset_repository.get_industrial_annotation_count(dataset_name)
        except (FileNotFoundError, ValueError) as exc:
            raise self._map_repository_error(exc) from exc
        if latest is not None:
            return self._to_response(latest)

        class_counts: dict[str, int] = {}
        for class_map in counts.values():
            for class_name, n in class_map.items():
                class_counts[class_name] = int(class_counts.get(class_name, 0) + int(n))
        class_labels = sorted(class_counts.keys())
        (
            task_type,
            classification_mode,
            label_source,
            segmentation_supported,
            segmentation_notes,
        ) = resolve_industrial_task_profile(
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
            raise InvalidRequestError("Industrial training produced no curve points.")

        has_annotation_blob = any(
            sample.annotation_blob is not None for sample in train_samples
        ) or any(sample.annotation_blob is not None for sample in test_samples)
        if has_annotation_blob:
            seg_iou_train, seg_dice_train, ann_train = self.segmentation_baseline.baseline_metrics(
                train_samples
            )
            seg_iou_test, seg_dice_test, ann_test = self.segmentation_baseline.baseline_metrics(
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
        ) = resolve_industrial_task_profile(
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
        try:
            sample = self.dataset_repository.get_industrial_sample(
                dataset_name=dataset_name,
                split=split,
                class_name=class_name,
                index=sample_index,
            )
            annotation_count = self.dataset_repository.get_industrial_annotation_count(dataset_name)
            class_counts = self.dataset_repository.get_industrial_counts().get(dataset_name, {})
        except (FileNotFoundError, ValueError) as exc:
            raise self._map_repository_error(exc) from exc
        class_labels = sorted(
            {class_name_key for split_map in class_counts.values() for class_name_key in split_map}
        )
        task_type, _, _, segmentation_supported, segmentation_notes = resolve_industrial_task_profile(
            dataset_name=dataset_name,
            class_labels=class_labels,
            annotated_samples=annotation_count,
        )

        mask = np.zeros(sample.image_rgb.shape[:2], dtype=np.uint8)
        bbox_count = 0
        source: Literal["annotation_xml", "none"] = "none"

        if sample.annotation_blob:
            parsed_mask, parsed_bbox_count = self.segmentation_baseline.mask_from_annotation(
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
