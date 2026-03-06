"""Tests for BUSI repository persistence and runtime normalization."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultrasound.api.config import AppConfig
from ultrasound.api.database.models import BusiSampleORM, BusiTrainingRunORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import BusiTrainingRunRecord
from ultrasound.api.repositories.busi_repository import BusiRepository
from ultrasound.data import create_sample_data


def _make_config(tmp_path: Path) -> AppConfig:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = project_root / "ui"
    artifacts_dir = project_root / "outputs" / "api"
    for path in (busi_dir, ndt_dir, ui_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project_root,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
        database_url=f"sqlite:///{(tmp_path / 'busi.sqlite3').resolve()}",
    )


def _png_bytes(fill_value: int, *, size: tuple[int, int] = (12, 16)) -> bytes:
    image = np.full((size[0], size[1], 3), fill_value, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_bytes(fill_value: int, *, size: tuple[int, int] = (6, 8)) -> bytes:
    mask = np.full(size, fill_value, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(mask, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _run_record(*, include_normal: bool) -> BusiTrainingRunRecord:
    return BusiTrainingRunRecord(
        created_at=datetime.now(tz=timezone.utc),
        include_normal=include_normal,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        train_samples=4,
        test_samples=2,
        class_counts={"benign": 2, "malignant": 2},
        class_labels=["benign", "malignant"],
        train_accuracy=0.8,
        test_accuracy=0.7,
        train_loss=0.2,
        test_loss=0.3,
    )


def test_busi_repository_sync_counts_and_sample_lookup(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    create_sample_data(str(config.busi_dir), num_samples=2)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = BusiRepository(config, db)

    assert repository.sync_busi_from_filesystem() == 4
    assert repository.sync_busi_from_filesystem() == 0
    assert repository.get_busi_counts() == {"benign": 2, "malignant": 2, "normal": 0}

    sample = repository.get_busi_sample("  BeNiGn  ", 3)
    assert sample.class_name == "benign"
    assert sample.total_samples == 2
    assert sample.resolved_index == 1
    assert sample.image_rgb.shape[:2] == sample.mask.shape

    with pytest.raises(FileNotFoundError, match="BUSI class 'unknown' not found"):
        repository.get_busi_sample("unknown", 0)
    with pytest.raises(ValueError, match="sample index must be >= 0"):
        repository.get_busi_sample("benign", -1)
    with pytest.raises(FileNotFoundError, match="No BUSI images found for class 'normal'"):
        repository.get_busi_sample("normal", 0)


def test_busi_repository_upload_validates_updates_and_resizes_masks(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = BusiRepository(config, db)

    with pytest.raises(ValueError, match="Invalid BUSI class"):
        repository.add_busi_uploaded_sample("weird", "train", "bad.png", _png_bytes(80))
    with pytest.raises(ValueError, match="BUSI split must be 'train' or 'test'"):
        repository.add_busi_uploaded_sample("benign", "dev", "bad.png", _png_bytes(80))

    uploaded = repository.add_busi_uploaded_sample(
        " Benign ",
        " TRAIN ",
        "../Uploaded Case.PNG",
        _png_bytes(90, size=(10, 14)),
        _mask_bytes(255, size=(4, 5)),
    )
    assert uploaded.class_name == "benign"
    assert uploaded.image_filename == "uploaded_case.png"
    assert uploaded.total_class_samples == 1

    sample = repository.get_busi_sample("benign", 0)
    assert sample.mask.shape == (10, 14)
    assert int(sample.mask.max()) == 255

    updated = repository.add_busi_uploaded_sample(
        "benign",
        "test",
        "uploaded case.png",
        _png_bytes(180, size=(10, 14)),
        None,
    )
    updated_sample = repository.get_busi_sample("benign", 0)

    assert updated.sample_id == uploaded.sample_id
    assert updated.total_class_samples == 1
    assert int(updated_sample.image_rgb.mean()) == 180
    assert int(updated_sample.mask.sum()) == 0


def test_busi_repository_lists_training_samples_and_skips_invalid_rows(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = BusiRepository(config, db)

    repository.add_busi_uploaded_sample("benign", "train", "benign.png", _png_bytes(60))
    repository.add_busi_uploaded_sample("malignant", "test", "malignant.png", _png_bytes(120))
    repository.add_busi_uploaded_sample("normal", "train", "normal.png", _png_bytes(200))

    with db.session_scope() as session:
        session.add(
            BusiSampleORM(
                class_name="mystery",
                image_filename="bad.png",
                sample_stem="bad",
                image_blob=_png_bytes(33),
                mask_blob=None,
                width=16,
                height=12,
                label=0,
                split="oops",
                source_hash="x" * 64,
            )
        )

    without_normal = repository.list_busi_training_samples(include_normal=False)
    with_normal = repository.list_busi_training_samples(include_normal=True)

    assert {(sample.class_name, sample.split) for sample in without_normal} == {
        ("benign", "train"),
        ("malignant", "test"),
    }
    assert {(sample.class_name, sample.split) for sample in with_normal} == {
        ("benign", "train"),
        ("malignant", "test"),
        ("normal", "train"),
    }


def test_busi_repository_returns_latest_valid_training_run(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseSessionManager(config.database_url)
    db.create_schema()
    repository = BusiRepository(config, db)

    first = repository.save_busi_training_run(_run_record(include_normal=False))
    repository.save_busi_training_run(_run_record(include_normal=True))

    with db.session_scope() as session:
        session.add(
            BusiTrainingRunORM(
                include_normal=False,
                train_accuracy=0.0,
                test_accuracy=0.0,
                payload_json="{bad json",
            )
        )

    latest_false = repository.get_latest_busi_training_run(include_normal=False)
    latest_true = repository.get_latest_busi_training_run(include_normal=True)

    assert latest_false is not None
    assert latest_false.run_id == first.run_id
    assert latest_true is not None
    assert latest_true.include_normal is True
