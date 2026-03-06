"""BUSI model training service backed by SQL-stored images."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

from ultrasound.api.models.domain import BusiTrainingCurvePointRecord, BusiTrainingRunRecord
from ultrasound.api.models.schemas import (
    BusiTrainingCurvePoint,
    BusiTrainingRequest,
    BusiTrainingResponse,
)
from ultrasound.api.services.interfaces import BusiTrainingRepository


class BusiTrainingService:
    """Runs lightweight BUSI training jobs and stores metrics/curves."""

    def __init__(self, dataset_repository: BusiTrainingRepository):
        self.dataset_repository = dataset_repository

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

    def _prepare_dataset(
        self, include_normal: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[str]]:
        samples = self.dataset_repository.list_busi_training_samples(include_normal=include_normal)
        if not samples:
            raise ValueError("No BUSI samples were found in SQL storage.")

        train_samples = [sample for sample in samples if sample.split == "train"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        if not train_samples or not test_samples:
            raise ValueError("BUSI SQL dataset must contain both train and test samples.")

        class_counts: dict[str, int] = {}
        for sample in samples:
            class_counts[sample.class_name] = int(class_counts.get(sample.class_name, 0) + 1)

        if include_normal and class_counts.get("normal", 0) <= 0:
            raise ValueError(
                "include_normal=True requires normal BUSI samples to be present in SQL storage."
            )

        train_labels = sorted({sample.label for sample in train_samples})
        test_labels = sorted({sample.label for sample in test_samples})
        if not set(test_labels).issubset(set(train_labels)):
            raise ValueError(
                "BUSI train split does not cover all classes present in test split. "
                "Re-sync SQL dataset."
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

        label_to_name = {0: "benign", 1: "malignant", 2: "normal"}
        class_labels = [label_to_name[label] for label in train_labels]
        return x_train, y_train, x_test, y_test, class_counts, class_labels

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
    ) -> list[BusiTrainingCurvePointRecord]:
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

        curve: list[BusiTrainingCurvePointRecord] = []
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
                BusiTrainingCurvePointRecord(
                    epoch=epoch,
                    train_accuracy=train_acc,
                    test_accuracy=test_acc,
                    train_loss=train_loss,
                    test_loss=test_loss,
                )
            )

        return curve

    def _to_response(self, run: BusiTrainingRunRecord) -> BusiTrainingResponse:
        return BusiTrainingResponse(
            run_id=run.run_id,
            generated_at=run.created_at,
            storage="sql",
            include_normal=run.include_normal,
            epochs=run.epochs,
            batch_size=run.batch_size,
            learning_rate=run.learning_rate,
            train_samples=run.train_samples,
            test_samples=run.test_samples,
            class_counts=run.class_counts,
            class_labels=run.class_labels,
            train_accuracy=run.train_accuracy,
            test_accuracy=run.test_accuracy,
            train_loss=run.train_loss,
            test_loss=run.test_loss,
            curve=[
                BusiTrainingCurvePoint(
                    epoch=point.epoch,
                    train_accuracy=point.train_accuracy,
                    test_accuracy=point.test_accuracy,
                    train_loss=point.train_loss,
                    test_loss=point.test_loss,
                )
                for point in run.curve
            ],
            notes=run.notes,
        )

    def get_latest_run(self, include_normal: bool = False) -> BusiTrainingResponse:
        latest = self.dataset_repository.get_latest_busi_training_run(include_normal=include_normal)
        if latest is not None:
            return self._to_response(latest)

        counts = self.dataset_repository.get_busi_counts()
        if include_normal:
            selected_counts = counts
            class_labels = ["benign", "malignant", "normal"]
        else:
            selected_counts = {k: v for k, v in counts.items() if k in {"benign", "malignant"}}
            class_labels = ["benign", "malignant"]

        return BusiTrainingResponse(
            run_id=None,
            generated_at=datetime.now(timezone.utc),
            storage="sql",
            include_normal=include_normal,
            epochs=0,
            batch_size=0,
            learning_rate=0.0,
            train_samples=0,
            test_samples=0,
            class_counts=selected_counts,
            class_labels=class_labels,
            train_accuracy=None,
            test_accuracy=None,
            train_loss=None,
            test_loss=None,
            curve=[],
            notes="No training run found. Trigger /datasets/busi/training/run first.",
        )

    def run_training(self, request: BusiTrainingRequest) -> BusiTrainingResponse:
        x_train, y_train, x_test, y_test, class_counts, class_labels = self._prepare_dataset(
            include_normal=request.include_normal
        )
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
            raise ValueError("Training produced no curve points.")

        last = curve[-1]
        notes = "Accuracy is measured on deterministic SQL train/test splits."
        run = BusiTrainingRunRecord(
            created_at=datetime.now(timezone.utc),
            include_normal=request.include_normal,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            train_samples=int(x_train.shape[0]),
            test_samples=int(x_test.shape[0]),
            class_counts=class_counts,
            class_labels=class_labels,
            train_accuracy=last.train_accuracy,
            test_accuracy=last.test_accuracy,
            train_loss=last.train_loss,
            test_loss=last.test_loss,
            curve=curve,
            notes=notes,
        )

        persisted = self.dataset_repository.save_busi_training_run(run)
        return self._to_response(persisted)
