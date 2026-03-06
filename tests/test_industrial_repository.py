"""Tests for industrial repository normalization and runtime behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.database.models import IndustrialTrainingRunORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import IndustrialTrainingRunRecord
from ultrasound.api.repositories.industrial_repository import IndustrialRepository


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = project_root / "ui"
    artifacts_dir = project_root / "artifacts"
    for path in (busi_dir, ndt_dir, ui_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project_root,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
        database_url=f"sqlite:///{(tmp_path / 'industrial.sqlite3').resolve()}",
    )


def _png_bytes(fill_value: int) -> bytes:
    image = np.full((12, 16, 3), fill_value, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_industrial_repository_normalizes_sample_lookup_inputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="Rolled In Scale",
        image_filename="Sample 01.png",
        image_blob=_png_bytes(120),
        annotation_blob=b"<xml/>",
    )

    sample = repository.get_industrial_sample(" Steel_Defect ", " TRAIN ", " rolled in scale ", 0)

    assert sample.dataset_name == "steel_defect"
    assert sample.split == "train"
    assert sample.class_name == "rolled_in_scale"
    assert sample.has_annotation is True


def test_industrial_repository_list_and_counts_accept_case_insensitive_dataset_name(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="train.png",
        image_blob=_png_bytes(100),
        annotation_blob=b"<xml/>",
    )
    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="test",
        class_name="crazing",
        image_filename="test.png",
        image_blob=_png_bytes(140),
        annotation_blob=None,
    )

    samples, class_counts, class_names = repository.list_industrial_training_samples(" STEEL_DEFECT ")

    assert len(samples) == 2
    assert class_counts == {"crazing": 2}
    assert class_names == ["crazing"]
    assert repository.get_industrial_annotation_count(" Steel_Defect ") == 1


def test_industrial_repository_updates_existing_uploaded_sample(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    first = repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="dup.png",
        image_blob=_png_bytes(60),
        annotation_blob=b"<xml/>",
    )
    second = repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="dup.png",
        image_blob=_png_bytes(180),
        annotation_blob=None,
    )

    sample = repository.get_industrial_sample("steel_defect", "train", "crazing", 0)

    assert first.sample_id == second.sample_id
    assert second.total_class_samples == 1
    assert int(sample.image_rgb.mean()) == 180
    assert sample.has_annotation is False


def test_industrial_repository_rebuilds_missing_test_split_for_training(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="train_a.png",
        image_blob=_png_bytes(80),
        annotation_blob=None,
    )
    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="train_b.png",
        image_blob=_png_bytes(120),
        annotation_blob=None,
    )

    samples, class_counts, class_names = repository.list_industrial_training_samples("steel_defect")

    assert sorted(sample.split for sample in samples) == ["test", "train"]
    assert class_counts == {"crazing": 2}
    assert class_names == ["crazing"]


def test_industrial_repository_rejects_missing_or_invalid_sample_requests(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    with pytest.raises(ValueError, match="No samples found"):
        repository.list_industrial_training_samples("steel_defect")

    repository.add_industrial_uploaded_sample(
        dataset_name="steel_defect",
        split="train",
        class_name="crazing",
        image_filename="train.png",
        image_blob=_png_bytes(90),
        annotation_blob=None,
    )

    with pytest.raises(FileNotFoundError, match="Invalid industrial dataset"):
        repository.get_industrial_sample("unknown", "train", "crazing", 0)
    with pytest.raises(FileNotFoundError, match="class_name must not be empty"):
        repository.get_industrial_sample("steel_defect", "train", "   ", 0)
    with pytest.raises(ValueError, match="sample index must be >= 0"):
        repository.get_industrial_sample("steel_defect", "train", "crazing", -1)
    with pytest.raises(FileNotFoundError, match="No industrial samples found"):
        repository.get_industrial_sample("steel_defect", "test", "crazing", 0)


def test_industrial_training_run_roundtrip_and_invalid_payload_fallback(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = IndustrialRepository(config, db)

    saved = repository.save_industrial_training_run(
        IndustrialTrainingRunRecord(
            created_at=datetime.now(tz=timezone.utc),
            dataset_name="steel_defect",
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            train_samples=2,
            test_samples=1,
            class_counts={"crazing": 3},
            class_labels=["crazing"],
            train_accuracy=0.9,
            test_accuracy=0.8,
            train_loss=0.2,
            test_loss=0.3,
        )
    )

    loaded = repository.get_latest_industrial_training_run("steel_defect")

    assert loaded is not None
    assert loaded.run_id == saved.run_id
    assert loaded.dataset_name == "steel_defect"

    with db.session_scope() as session:
        session.add(
            IndustrialTrainingRunORM(
                dataset_name="steel_defect",
                train_accuracy=0.0,
                test_accuracy=0.0,
                payload_json="{bad json",
            )
        )

    assert repository.get_latest_industrial_training_run("steel_defect") is None
