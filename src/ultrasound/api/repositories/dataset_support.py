"""Shared support code for dataset-oriented repositories."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.database.models import DatasetMetaORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import NdtDefectRecord


class DatasetRepositorySupport:
    """Common helpers reused across BUSI, NDT, and industrial repositories."""

    CLASSES = ("benign", "malignant", "normal")
    CLASS_TO_LABEL = {"benign": 0, "malignant": 1, "normal": 2}
    INDUSTRIAL_DATASETS = ("steel_defect", "neu_surface", "casting_defect")
    INDUSTRIAL_SPLITS = {
        "steel_defect": {"train", "valid", "test"},
        "neu_surface": {"train", "validation"},
        "casting_defect": {"train", "test", "full"},
    }

    def __init__(self, config: AppConfig, db: DatabaseSessionManager) -> None:
        self.config = config
        self.db = db

    def _to_float_scalar(self, value: Any, default: float) -> float:
        try:
            arr = np.asarray(value)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    def _array_to_blob(self, arr: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, np.asarray(arr, dtype=np.float64), allow_pickle=False)
        return buffer.getvalue()

    def _blob_to_array(self, blob: bytes) -> np.ndarray:
        buffer = BytesIO(blob)
        arr = np.load(buffer, allow_pickle=False)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    def _canonical_png_rgb(self, image_path: Path) -> tuple[bytes, int, int]:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
            width, height = image_rgb.size
            buffer = BytesIO()
            image_rgb.save(buffer, format="PNG")
        return buffer.getvalue(), int(width), int(height)

    def _canonical_png_mask(self, mask_path: Path | None, width: int, height: int) -> bytes | None:
        if mask_path is None:
            return None
        with Image.open(mask_path) as pil_mask:
            mask_gray = pil_mask.convert("L")
            if mask_gray.size != (width, height):
                mask_gray = mask_gray.resize((width, height), Image.Resampling.NEAREST)
            buffer = BytesIO()
            mask_gray.save(buffer, format="PNG")
        return buffer.getvalue()

    def _canonical_png_rgb_bytes(self, image_blob: bytes) -> tuple[bytes, int, int]:
        with Image.open(BytesIO(image_blob)) as pil_image:
            image_rgb = pil_image.convert("RGB")
            width, height = image_rgb.size
            buffer = BytesIO()
            image_rgb.save(buffer, format="PNG")
        return buffer.getvalue(), int(width), int(height)

    def _canonical_png_mask_bytes(
        self, mask_blob: bytes | None, width: int, height: int
    ) -> bytes | None:
        if mask_blob is None:
            return None
        with Image.open(BytesIO(mask_blob)) as pil_mask:
            mask_gray = pil_mask.convert("L")
            if mask_gray.size != (width, height):
                mask_gray = mask_gray.resize((width, height), Image.Resampling.NEAREST)
            buffer = BytesIO()
            mask_gray.save(buffer, format="PNG")
        return buffer.getvalue()

    def _safe_filename(self, filename: str, default_stem: str) -> str:
        stem = Path(filename).stem if filename else ""
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
        if not stem:
            stem = default_stem
        return f"{stem}.png"

    def _decode_rgb_blob(self, blob: bytes) -> np.ndarray:
        with Image.open(BytesIO(blob)) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        return image

    def _decode_mask_blob(self, blob: bytes | None, shape: tuple[int, int]) -> np.ndarray:
        if blob is None:
            return np.zeros(shape, dtype=np.uint8)
        with Image.open(BytesIO(blob)) as pil_mask:
            mask = np.asarray(pil_mask.convert("L"), dtype=np.uint8)
        if mask.shape != shape:
            resized = Image.fromarray(mask, mode="L").resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
            mask = np.asarray(resized, dtype=np.uint8)
        return mask

    @staticmethod
    def _resolve_train_cutoff(n_samples: int) -> int:
        if n_samples <= 1:
            return n_samples
        train_count = int(round(0.8 * float(n_samples)))
        train_count = max(1, min(train_count, n_samples - 1))
        return train_count

    def _industrial_roots(self) -> dict[str, Path]:
        return {
            "steel_defect": self.config.data_dir / "steel_defect",
            "neu_surface": self.config.data_dir / "neu_surface",
            "casting_defect": self.config.data_dir / "casting_defect",
        }

    def _collect_industrial_sources(self) -> list[tuple[str, str, str, Path, Path | None]]:
        records: list[tuple[str, str, str, Path, Path | None]] = []
        roots = self._industrial_roots()

        steel_root = roots["steel_defect"] / "NEU Metal Surface Defects Data"
        if steel_root.exists():
            for split in ("train", "valid", "test"):
                split_dir = steel_root / split
                if not split_dir.exists():
                    continue
                for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*.bmp")):
                        records.append(("steel_defect", split, class_dir.name.lower(), image_path, None))

        neu_root = roots["neu_surface"] / "NEU-DET"
        if neu_root.exists():
            for split in ("train", "validation"):
                images_root = neu_root / split / "images"
                ann_root = neu_root / split / "annotations"
                if not images_root.exists():
                    continue
                for class_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*")):
                        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                            continue
                        annotation_path = ann_root / f"{image_path.stem}.xml"
                        records.append(
                            (
                                "neu_surface",
                                split,
                                class_dir.name.lower(),
                                image_path,
                                annotation_path if annotation_path.exists() else None,
                            )
                        )

        casting_train_test_root = self.config.data_dir / "casting_defect" / "casting_data" / "casting_data"
        if casting_train_test_root.exists():
            for split in ("train", "test"):
                split_dir = casting_train_test_root / split
                if not split_dir.exists():
                    continue
                for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    for image_path in sorted(class_dir.glob("*.jpeg")):
                        records.append(("casting_defect", split, class_dir.name.lower(), image_path, None))

        casting_flat_root = self.config.data_dir / "casting_defect" / "casting_512x512" / "casting_512x512"
        if casting_flat_root.exists():
            for class_dir in sorted(path for path in casting_flat_root.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.glob("*.jpeg")):
                    records.append(("casting_defect", "full", class_dir.name.lower(), image_path, None))

        return records

    def _meta_get(self, key: str) -> str | None:
        with self.db.session_scope() as session:
            row = session.get(DatasetMetaORM, key)
            return row.value if row is not None else None

    @staticmethod
    def _set_meta_value(session: Any, key: str, value: str) -> None:
        row = session.get(DatasetMetaORM, key)
        if row is None:
            session.add(DatasetMetaORM(key=key, value=value))
        else:
            row.value = value

    def _compute_busi_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for class_name in self.CLASSES:
            class_dir = self.config.busi_dir / class_name
            if not class_dir.exists():
                digest.update(f"{class_name}:missing|".encode())
                continue
            image_paths = sorted(path for path in class_dir.glob("*.png") if "_mask" not in path.stem)
            digest.update(f"{class_name}:{len(image_paths)}|".encode())
            for image_path in image_paths:
                stat = image_path.stat()
                digest.update(f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
                mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                if mask_candidates:
                    mask_stat = mask_candidates[0].stat()
                    digest.update(
                        f"{mask_candidates[0].name}:{mask_stat.st_size}:{mask_stat.st_mtime_ns}|".encode()
                    )
        return digest.hexdigest()

    def _compute_ndt_fingerprint(self) -> str:
        digest = hashlib.sha256()
        if not self.config.ndt_dir.exists():
            digest.update(b"ndt:missing")
            return digest.hexdigest()

        sample_paths = sorted(self.config.ndt_dir.glob("*.npz"))
        digest.update(f"ndt:{len(sample_paths)}|".encode())
        for sample_path in sample_paths:
            stat = sample_path.stat()
            digest.update(f"{sample_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
        return digest.hexdigest()

    def _compute_industrial_fingerprint(self) -> str:
        digest = hashlib.sha256()
        sources = self._collect_industrial_sources()
        digest.update(f"industrial:{len(sources)}|".encode())
        for dataset_name, split, class_name, image_path, annotation_path in sources:
            image_stat = image_path.stat()
            digest.update(
                (
                    f"{dataset_name}:{split}:{class_name}:{image_path.name}:"
                    f"{image_stat.st_size}:{image_stat.st_mtime_ns}|"
                ).encode()
            )
            if annotation_path is not None:
                ann_stat = annotation_path.stat()
                digest.update((f"{annotation_path.name}:{ann_stat.st_size}:{ann_stat.st_mtime_ns}|").encode())
        return digest.hexdigest()

    def _build_defect_records(self, defects_obj: Any) -> list[NdtDefectRecord]:
        arr = np.asarray(defects_obj)

        if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
            records: list[NdtDefectRecord] = []
            for row in arr:
                depth = float(row[0]) if np.isfinite(row[0]) else None
                amp = float(row[1]) if np.isfinite(row[1]) else None
                if depth is not None or amp is not None:
                    records.append(NdtDefectRecord(depth_m=depth, amplitude=amp))
            return records

        if arr.size == 0:
            return []

        try:
            defects_raw = arr.tolist()
            if not isinstance(defects_raw, list):
                defects_raw = [defects_raw]
        except Exception:
            return []

        defects: list[NdtDefectRecord] = []
        for item in defects_raw:
            if isinstance(item, dict):
                record = NdtDefectRecord(depth_m=item.get("depth_m"), amplitude=item.get("amplitude"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                record = NdtDefectRecord(depth_m=item[0], amplitude=item[1])
            else:
                continue

            if record.depth_m is not None or record.amplitude is not None:
                defects.append(record)
        return defects
