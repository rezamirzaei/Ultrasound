"""Field-style image record storage for YOLO workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import ValidationError

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    FieldYoloLabel,
    FieldYoloMetadata,
    FieldYoloRecordDetail,
    FieldYoloRecordManifest,
    FieldYoloRecordSummary,
)
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.yolo_utils import parse_yolo_txt_labels


class FieldYoloService:
    """Persist and retrieve field inspection-style images + metadata for YOLO usage."""

    def __init__(self, config: AppConfig, media_service: MediaService):
        self.config = config
        self.media_service = media_service
        self.records_root = self.config.artifacts_dir / "field_yolo" / "records"
        self.records_root.mkdir(parents=True, exist_ok=True)

    def _record_dir(self, record_id: str) -> Path:
        return self.records_root / record_id

    def _manifest_path(self, record_id: str) -> Path:
        return self._record_dir(record_id) / "manifest.json"

    def _load_manifest(self, record_id: str) -> FieldYoloRecordManifest:
        manifest_path = self._manifest_path(record_id)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Field record '{record_id}' not found")
        return FieldYoloRecordManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    def _canonical_png(self, image_blob: bytes) -> tuple[bytes, int, int]:
        try:
            with Image.open(BytesIO(image_blob)) as pil_image:
                image_rgb = pil_image.convert("RGB")
                width, height = image_rgb.size
                buffer = BytesIO()
                image_rgb.save(buffer, format="PNG")
        except Exception as exc:
            raise ValueError("Invalid image payload: expected a readable image file") from exc
        return buffer.getvalue(), int(width), int(height)

    def _parse_labels(self, metadata: FieldYoloMetadata, labels_text: str) -> list[FieldYoloLabel]:
        return parse_yolo_txt_labels(labels_text, class_names=list(metadata.class_names or []))

    def create_record(
        self,
        metadata: FieldYoloMetadata,
        image_filename: str,
        image_blob: bytes,
        labels_filename: str | None = None,
        labels_blob: bytes | None = None,
    ) -> FieldYoloRecordSummary:
        record_id = uuid4().hex
        created_at = datetime.now(tz=timezone.utc)
        record_dir = self._record_dir(record_id)
        record_dir.mkdir(parents=True, exist_ok=True)

        png_blob, width, height = self._canonical_png(image_blob)
        (record_dir / "image.png").write_bytes(png_blob)

        parsed_labels: list[FieldYoloLabel] = []
        stored_labels_filename = None
        if labels_blob:
            labels_text = labels_blob.decode("utf-8", errors="ignore")
            parsed_labels = self._parse_labels(metadata, labels_text)
            (record_dir / "labels.txt").write_text(labels_text.strip() + "\n", encoding="utf-8")
            stored_labels_filename = labels_filename or "labels.txt"

        manifest = FieldYoloRecordManifest(
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            image_filename=image_filename or "image.png",
            stored_image="image.png",
            labels_filename=stored_labels_filename,
            stored_labels="labels.txt" if labels_blob else None,
            width=width,
            height=height,
        )
        self._manifest_path(record_id).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return FieldYoloRecordSummary(
            record_id=record_id,
            created_at=created_at,
            asset_id=metadata.asset_id,
            location_name=metadata.location_name,
            captured_at=metadata.captured_at,
            has_labels=bool(parsed_labels),
            width=width,
            height=height,
        )

    def list_records(self, limit: int = 50) -> list[FieldYoloRecordSummary]:
        limit = max(1, min(int(limit), 200))
        summaries: list[FieldYoloRecordSummary] = []

        record_dirs = sorted(
            (path for path in self.records_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for record_dir in record_dirs:
            if len(summaries) >= limit:
                break
            record_id = record_dir.name
            try:
                manifest = self._load_manifest(record_id)
            except (FileNotFoundError, ValidationError, json.JSONDecodeError):
                continue

            summaries.append(
                FieldYoloRecordSummary(
                    record_id=manifest.record_id,
                    created_at=manifest.created_at,
                    asset_id=manifest.metadata.asset_id,
                    location_name=manifest.metadata.location_name,
                    captured_at=manifest.metadata.captured_at,
                    has_labels=bool(manifest.stored_labels),
                    width=manifest.width,
                    height=manifest.height,
                )
            )

        summaries.sort(key=lambda item: item.created_at, reverse=True)
        return summaries

    def get_record(self, record_id: str) -> FieldYoloRecordDetail:
        manifest = self._load_manifest(record_id)
        record_dir = self._record_dir(record_id)
        image_blob = (record_dir / manifest.stored_image).read_bytes()

        labels: list[FieldYoloLabel] = []
        raw_labels = None
        if manifest.stored_labels:
            raw_labels = (record_dir / manifest.stored_labels).read_text(encoding="utf-8")
            labels = self._parse_labels(manifest.metadata, raw_labels)

        try:
            with Image.open(BytesIO(image_blob)) as pil_image:
                image_rgb = pil_image.convert("RGB")
                image_arr = np.asarray(image_rgb, dtype=np.uint8)
        except Exception as exc:
            raise ValueError("Stored image file is unreadable") from exc

        return FieldYoloRecordDetail(
            record_id=manifest.record_id,
            created_at=manifest.created_at,
            metadata=manifest.metadata,
            image_filename=manifest.image_filename,
            width=manifest.width,
            height=manifest.height,
            image_data_url=self.media_service.as_png_data_url(image_arr),
            labels=labels,
            raw_labels=raw_labels,
        )

    def load_image_rgb(self, record_id: str) -> np.ndarray:
        """Load stored image as RGB numpy array for model inference."""
        manifest = self._load_manifest(record_id)
        record_dir = self._record_dir(record_id)
        image_blob = (record_dir / manifest.stored_image).read_bytes()
        try:
            with Image.open(BytesIO(image_blob)) as pil_image:
                image_rgb = pil_image.convert("RGB")
                return np.asarray(image_rgb, dtype=np.uint8)
        except Exception as exc:
            raise ValueError("Stored image file is unreadable") from exc
