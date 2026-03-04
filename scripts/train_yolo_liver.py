#!/usr/bin/env python3
"""Train YOLO for Liver Ultrasound Detection.

End-to-end script that:
  1. Prepares / downloads the dataset (or creates synthetic data).
  2. Converts annotations to YOLO format.
  3. Fine-tunes a pretrained YOLO model.
  4. Reports metrics and saves weights.

Usage examples:
  # With real Kaggle data (requires kaggle credentials):
  python scripts/train_yolo_liver.py

  # With synthetic demo data (no credentials needed):
  python scripts/train_yolo_liver.py --synthetic --epochs 5

  # Custom settings:
  python scripts/train_yolo_liver.py --synthetic --epochs 10 --batch 8 --imgsz 416
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project ``src/`` is importable when running as a script.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from ultrasound.api.services.liver_dataset import (
    CLASS_NAMES,
    create_synthetic_liver_dataset,
    download_liver_dataset,
    resolve_liver_paths,
    summarize_dataset,
)
from ultrasound.api.services.yolo_trainer import (
    YoloDatasetPreparer,
    YoloTrainer,
    YoloTrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_yolo_liver")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the Liver Ultrasound Detection dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_project_root / "data" / "liver_ultrasound_detection"),
        help="Path to raw dataset (images + train.csv)",
    )
    parser.add_argument(
        "--yolo-dir",
        type=str,
        default=str(_project_root / "data" / "liver_yolo"),
        help="Output directory for YOLO-formatted dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_project_root / "outputs" / "yolo_runs"),
        help="Directory for training run outputs",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--n-samples", type=int, default=30, help="Synthetic samples count")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device ('' = auto)")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze N backbone layers")
    parser.add_argument("--run-name", type=str, default="liver_detection", help="Run name")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    yolo_dir = Path(args.yolo_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Get the dataset
    logger.info("=" * 60)
    logger.info("Step 1: Preparing raw dataset")
    logger.info("=" * 60)

    if args.synthetic:
        paths = create_synthetic_liver_dataset(data_dir, n_samples=args.n_samples)
    else:
        paths = resolve_liver_paths(data_dir.parent)
        if not paths.is_ready:
            logger.info("Dataset not found locally, attempting Kaggle download ...")
            paths = download_liver_dataset(data_dir)

    summary = summarize_dataset(paths)
    logger.info("Dataset summary: %s", json.dumps(summary, indent=2, default=str))

    # Step 2: Convert to YOLO format
    logger.info("=" * 60)
    logger.info("Step 2: Converting to YOLO format")
    logger.info("=" * 60)

    preparer = YoloDatasetPreparer(
        source_images_dir=paths.train_images_dir,
        annotations_csv=paths.annotations_csv,
        output_dir=yolo_dir,
        class_names=CLASS_NAMES,
        train_ratio=args.train_ratio,
    )
    data_yaml = preparer.prepare()
    logger.info("YOLO dataset ready: %s", data_yaml)

    # Step 3: Train
    logger.info("=" * 60)
    logger.info("Step 3: Training YOLO model")
    logger.info("=" * 60)

    config = YoloTrainingConfig(
        dataset_yaml=data_yaml,
        pretrained_weights=args.weights,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device,
        project_dir=output_dir,
        run_name=args.run_name,
        freeze_layers=args.freeze,
    )

    trainer = YoloTrainer()
    result = trainer.train(config)

    # Step 4: Report results
    logger.info("=" * 60)
    logger.info("Step 4: Training complete!")
    logger.info("=" * 60)
    logger.info("Run directory:  %s", result.run_dir)
    logger.info("Best weights:   %s", result.best_weights)
    logger.info("Last weights:   %s", result.last_weights)
    logger.info("Epochs trained: %d", result.epochs_completed)

    if result.metrics:
        logger.info("Metrics:")
        for key, value in sorted(result.metrics.items()):
            logger.info("  %s: %.4f", key, value)

    # Optional: Validate with best weights
    if result.best_weights and result.best_weights.exists():
        logger.info("Running final validation with best weights ...")
        val_metrics = trainer.validate(result.best_weights, data_yaml, image_size=args.imgsz)
        if val_metrics:
            logger.info("Validation metrics:")
            for key, value in sorted(val_metrics.items()):
                logger.info("  %s: %.4f", key, value)

    print("\n✓ YOLO liver detection training pipeline complete.")
    print(f"  Best weights: {result.best_weights}")
    print(f"  Run dir:      {result.run_dir}")


if __name__ == "__main__":
    main()

