"""SQLite-backed persistence for BUSI images and training artifacts."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from ultrasound.api.models.domain import (
    BusiSampleRecord,
    BusiTrainingRunRecord,
    BusiTrainingSampleRecord,
)


class BusiSqlRepository:
    """Persists BUSI image/mask data and training runs in SQLite."""

    CLASSES = ("benign", "malignant", "normal")
    CLASS_TO_LABEL = {"benign": 0, "malignant": 1, "normal": 2}

    def __init__(self, db_path: Path, busi_dir: Path):
        self.db_path = db_path
        self.busi_dir = busi_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self.sync_from_filesystem()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS dataset_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS busi_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    sample_stem TEXT NOT NULL,
                    image_blob BLOB NOT NULL,
                    mask_blob BLOB,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    label INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(class_name, image_filename)
                );

                CREATE INDEX IF NOT EXISTS idx_busi_samples_class_split
                ON busi_samples(class_name, split);

                CREATE TABLE IF NOT EXISTS busi_training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    include_normal INTEGER NOT NULL,
                    train_accuracy REAL,
                    test_accuracy REAL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_busi_training_runs_scope
                ON busi_training_runs(include_normal, id DESC);
                """)

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

    def _resolve_train_cutoff(self, n_samples: int) -> int:
        if n_samples <= 1:
            return n_samples
        train_count = int(round(0.8 * float(n_samples)))
        train_count = max(1, min(train_count, n_samples - 1))
        return train_count

    def _compute_filesystem_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for class_name in self.CLASSES:
            class_dir = self.busi_dir / class_name
            if not class_dir.exists():
                digest.update(f"{class_name}:missing|".encode("utf-8"))
                continue

            image_paths = sorted(
                path for path in class_dir.glob("*.png") if "_mask" not in path.stem
            )
            digest.update(f"{class_name}:{len(image_paths)}|".encode("utf-8"))

            for image_path in image_paths:
                stat = image_path.stat()
                digest.update(
                    f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode("utf-8")
                )
                mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                if mask_candidates:
                    mask_stat = mask_candidates[0].stat()
                    digest.update(
                        f"{mask_candidates[0].name}:{mask_stat.st_size}:{mask_stat.st_mtime_ns}|".encode(
                            "utf-8"
                        )
                    )
        return digest.hexdigest()

    def _read_fingerprint(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT value FROM dataset_meta WHERE key = ?",
            ("busi_fingerprint",),
        ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _write_fingerprint(self, conn: sqlite3.Connection, fingerprint: str) -> None:
        conn.execute(
            """
            INSERT INTO dataset_meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("busi_fingerprint", fingerprint),
        )

    def sync_from_filesystem(self) -> int:
        """Load BUSI files into SQL when source files changed."""
        fingerprint = self._compute_filesystem_fingerprint()

        with self._connect() as conn:
            current = self._read_fingerprint(conn)
            if current == fingerprint:
                return 0

            rows_to_insert: list[tuple[object, ...]] = []
            now_iso = datetime.now(timezone.utc).isoformat()

            for class_name in self.CLASSES:
                class_dir = self.busi_dir / class_name
                if not class_dir.exists():
                    continue

                image_paths = sorted(
                    path for path in class_dir.glob("*.png") if "_mask" not in path.stem
                )
                train_cutoff = self._resolve_train_cutoff(len(image_paths))

                for index, image_path in enumerate(image_paths):
                    image_blob, width, height = self._canonical_png_rgb(image_path)
                    mask_candidates = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                    mask_blob = self._canonical_png_mask(
                        mask_candidates[0] if mask_candidates else None,
                        width=width,
                        height=height,
                    )
                    source_hash = hashlib.sha256(image_blob + (mask_blob or b"")).hexdigest()
                    split = "train" if index < train_cutoff else "test"
                    rows_to_insert.append(
                        (
                            class_name,
                            image_path.name,
                            image_path.stem,
                            image_blob,
                            mask_blob,
                            width,
                            height,
                            self.CLASS_TO_LABEL[class_name],
                            split,
                            source_hash,
                            now_iso,
                        )
                    )

            conn.execute("DELETE FROM busi_samples")
            if rows_to_insert:
                conn.executemany(
                    """
                    INSERT INTO busi_samples(
                        class_name,
                        image_filename,
                        sample_stem,
                        image_blob,
                        mask_blob,
                        width,
                        height,
                        label,
                        split,
                        source_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )

            self._write_fingerprint(conn, fingerprint)
            return len(rows_to_insert)

    def get_busi_counts(self) -> dict[str, int]:
        self.sync_from_filesystem()
        counts = {name: 0 for name in self.CLASSES}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT class_name, COUNT(*) AS n FROM busi_samples GROUP BY class_name"
            ).fetchall()
        for row in rows:
            class_name = str(row["class_name"])
            if class_name in counts:
                counts[class_name] = int(row["n"])
        return counts

    def get_busi_sample(self, class_name: str, index: int = 0) -> BusiSampleRecord:
        self.sync_from_filesystem()

        if class_name not in self.CLASSES:
            raise FileNotFoundError(
                f"BUSI class '{class_name}' not found. Available classes: {self.CLASSES}"
            )
        if index < 0:
            raise ValueError("sample index must be >= 0")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM busi_samples WHERE class_name = ?",
                (class_name,),
            ).fetchone()
            total_samples = int(row["n"]) if row else 0
            if total_samples <= 0:
                raise FileNotFoundError(
                    f"No BUSI images found for class '{class_name}' in SQL storage."
                )

            resolved_index = int(index % total_samples)
            sample_row = conn.execute(
                """
                SELECT id, image_filename, image_blob, mask_blob
                FROM busi_samples
                WHERE class_name = ?
                ORDER BY image_filename
                LIMIT 1 OFFSET ?
                """,
                (class_name, resolved_index),
            ).fetchone()

        if sample_row is None:
            raise FileNotFoundError(
                f"Could not fetch BUSI sample for class '{class_name}' at index {index}."
            )

        image_rgb = self._decode_rgb_blob(bytes(sample_row["image_blob"]))
        mask = self._decode_mask_blob(
            bytes(sample_row["mask_blob"]) if sample_row["mask_blob"] is not None else None,
            shape=(int(image_rgb.shape[0]), int(image_rgb.shape[1])),
        )

        return BusiSampleRecord(
            class_name=class_name,
            requested_index=int(index),
            resolved_index=resolved_index,
            total_samples=total_samples,
            image_path=self.busi_dir / class_name / str(sample_row["image_filename"]),
            image_rgb=image_rgb,
            mask=mask,
        )

    def list_busi_training_samples(
        self, include_normal: bool = False
    ) -> list[BusiTrainingSampleRecord]:
        self.sync_from_filesystem()
        classes = self.CLASSES if include_normal else ("benign", "malignant")
        placeholders = ",".join("?" for _ in classes)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, class_name, label, split, image_blob
                FROM busi_samples
                WHERE class_name IN ({placeholders})
                ORDER BY class_name, image_filename
                """,
                classes,
            ).fetchall()

        samples: list[BusiTrainingSampleRecord] = []
        for row in rows:
            image_rgb = self._decode_rgb_blob(bytes(row["image_blob"]))
            class_name = str(row["class_name"])
            split = str(row["split"])
            if class_name not in self.CLASSES:
                continue
            if split not in {"train", "test"}:
                continue
            samples.append(
                BusiTrainingSampleRecord(
                    sample_id=int(row["id"]),
                    class_name=cast(Literal["benign", "malignant", "normal"], class_name),
                    label=int(row["label"]),
                    split=cast(Literal["train", "test"], split),
                    image_rgb=image_rgb,
                )
            )
        return samples

    def save_busi_training_run(self, run: BusiTrainingRunRecord) -> BusiTrainingRunRecord:
        payload = run.model_dump_json()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO busi_training_runs(
                    created_at,
                    include_normal,
                    train_accuracy,
                    test_accuracy,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.created_at.isoformat(),
                    1 if run.include_normal else 0,
                    float(run.train_accuracy),
                    float(run.test_accuracy),
                    payload,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Could not persist BUSI training run.")
            run_id = int(cursor.lastrowid)

        return run.model_copy(update={"run_id": run_id})

    def get_latest_busi_training_run(
        self, include_normal: bool = False
    ) -> BusiTrainingRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, payload_json
                FROM busi_training_runs
                WHERE include_normal = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (1 if include_normal else 0,),
            ).fetchone()

        if row is None:
            return None

        try:
            parsed = BusiTrainingRunRecord.model_validate_json(str(row["payload_json"]))
        except Exception:
            return None
        return parsed.model_copy(update={"run_id": int(row["id"])})
