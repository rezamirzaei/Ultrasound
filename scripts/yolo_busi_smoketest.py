#!/usr/bin/env python3
"""Download a BUSI-trained YOLO model and run a small inference smoke test.

This script is intentionally lightweight and uses the same services as the REST API.
It downloads a public BUSI YOLOv8 segmentation checkpoint and runs inference on a
couple of BUSI samples, saving annotated PNGs under outputs/api/.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.models.schemas import YoloPredictRequest


def _write_data_url_png(data_url: str, path: Path) -> None:
    header, b64 = data_url.split(",", 1)
    if "base64" not in header:
        raise ValueError("Expected base64 image data URL")
    path.write_bytes(base64.b64decode(b64))


def main() -> None:
    parser = argparse.ArgumentParser(description="BUSI YOLO smoke test (download + predict)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download weights")
    parser.add_argument(
        "--class",
        dest="class_name",
        default="benign",
        choices=["benign", "malignant", "normal"],
        help="BUSI class to test",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index (wraps mod N)")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="YOLO confidence threshold"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    container = ApplicationContainer()

    model_status = container.busi_yolo_lab_service.download_recommended_model(
        force=bool(args.force_download)
    )
    model_path = model_status.local_path
    if not model_status.downloaded:
        raise SystemExit("Model download failed (weights still missing).")

    request = YoloPredictRequest(
        model=model_path,
        confidence=float(args.confidence),
        iou_threshold=0.45,
        image_size=int(args.imgsz),
        max_detections=50,
    )

    sample = container.dataset_repository.get_busi_sample(args.class_name, int(args.index))
    prediction = container.yolo_service.predict(sample.image_rgb, request)

    out_dir = Path(container.config.artifacts_dir) / "ultrasound_yolo" / "busi" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.class_name}_{sample.resolved_index}_annotated.png"
    _write_data_url_png(prediction.annotated_image_data_url, out_path)

    print(f"Model: {prediction.model}")
    print(f"Sample: {args.class_name} index={args.index} resolved={sample.resolved_index}")
    print(f"Detections: {len(prediction.detections)}")
    print(f"Annotated output: {out_path}")


if __name__ == "__main__":
    main()

