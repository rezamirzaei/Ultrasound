"""Tests for YOLO training pipeline and liver dataset utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultrasound.api.models.schemas import YoloLabel, YoloXyxyBox
from ultrasound.api.services.yolo_trainer import YoloDatasetPreparer, YoloTrainingConfig
from ultrasound.api.services.yolo_utils import (
    format_yolo_labels,
    mask_to_xyxy,
    parse_yolo_txt_labels,
    xyxy_to_yolo_label,
)
from ultrasound.data.liver_dataset import (
    CLASS_NAMES,
    LiverDatasetPaths,
    _ensure_annotations_csv,
    create_synthetic_liver_dataset,
    load_annotations_csv,
    resolve_liver_paths,
    summarize_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dataset(tmp_path: Path) -> LiverDatasetPaths:
    """Create a small synthetic dataset for testing."""
    return create_synthetic_liver_dataset(tmp_path / "liver", n_samples=6)


@pytest.fixture
def mini_csv_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal images + CSV dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    csv_path = tmp_path / "train.csv"
    rows = []
    for i in range(4):
        img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        name = f"img_{i:03d}"
        Image.fromarray(img).save(images_dir / f"{name}.png")
        rows.append({
            "image_id": name,
            "x_min": "10",
            "y_min": "15",
            "x_max": "90",
            "y_max": "80",
        })

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max"])
        writer.writeheader()
        writer.writerows(rows)

    return images_dir, csv_path, tmp_path


# ---------------------------------------------------------------------------
# liver_dataset tests
# ---------------------------------------------------------------------------

class TestLiverDataset:
    def test_create_synthetic_dataset(self, tmp_dataset: LiverDatasetPaths) -> None:
        assert tmp_dataset.train_images_dir.is_dir()
        assert tmp_dataset.annotations_csv.is_file()
        train_images = list(tmp_dataset.train_images_dir.glob("*.png"))
        assert len(train_images) == 6

    def test_load_annotations_csv(self, tmp_dataset: LiverDatasetPaths) -> None:
        annotations = load_annotations_csv(tmp_dataset.annotations_csv)
        assert len(annotations) == 6
        for image_id, boxes in annotations.items():
            assert len(boxes) >= 1
            for box in boxes:
                assert "x_min" in box
                assert "y_min" in box
                assert "x_max" in box
                assert "y_max" in box
                assert box["x_max"] > box["x_min"]
                assert box["y_max"] > box["y_min"]

    def test_summarize_dataset(self, tmp_dataset: LiverDatasetPaths) -> None:
        summary = summarize_dataset(tmp_dataset)
        assert summary["images"] == 6
        assert summary["annotated_images"] == 6
        assert summary["total_boxes"] == 6

    def test_resolve_liver_paths(self, tmp_path: Path) -> None:
        paths = resolve_liver_paths(tmp_path)
        assert paths.root == tmp_path / "liver_ultrasound_detection"
        assert not paths.is_ready  # No data yet

    def test_class_names(self) -> None:
        assert CLASS_NAMES == ["liver", "mass"]

    def test_build_annotations_csv_supports_polygon_dict_payloads(self, tmp_path: Path) -> None:
        root = tmp_path / "liver"
        image_dir = root / "Benign" / "Benign" / "image"
        liver_dir = root / "Benign" / "Benign" / "segmentation" / "liver"
        mass_dir = root / "Benign" / "Benign" / "segmentation" / "mass"
        liver_dir.mkdir(parents=True, exist_ok=True)
        mass_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_dir / "case_001.png")
        (liver_dir / "case_001.json").write_text(
            json.dumps({"points": [[4, 5], [20, 5], [20, 24], [4, 24]]}),
            encoding="utf-8",
        )
        (mass_dir / "case_001.json").write_text(
            json.dumps({"points": [["bad", 1], [2, 3]]}),
            encoding="utf-8",
        )

        csv_path = _ensure_annotations_csv(LiverDatasetPaths(root=root))
        annotations = load_annotations_csv(csv_path)

        assert list(annotations) == ["Benign_case_001"]
        assert len(annotations["Benign_case_001"]) == 1
        assert annotations["Benign_case_001"][0]["class_id"] == 0

    def test_load_annotations_csv_skips_invalid_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "annotations.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "ok",
                    "x_min": "1",
                    "y_min": "2",
                    "x_max": "10",
                    "y_max": "12",
                    "class_id": "0",
                }
            )
            writer.writerow(
                {
                    "image_id": "bad_bounds",
                    "x_min": "9",
                    "y_min": "2",
                    "x_max": "5",
                    "y_max": "12",
                    "class_id": "0",
                }
            )
            writer.writerow(
                {
                    "image_id": "",
                    "x_min": "1",
                    "y_min": "2",
                    "x_max": "3",
                    "y_max": "4",
                    "class_id": "0",
                }
            )

        annotations = load_annotations_csv(csv_path)

        assert list(annotations) == ["ok"]


# ---------------------------------------------------------------------------
# YoloDatasetPreparer tests
# ---------------------------------------------------------------------------

class TestYoloDatasetPreparer:
    def test_prepare_creates_yolo_structure(
        self, mini_csv_dataset: tuple[Path, Path, Path]
    ) -> None:
        images_dir, csv_path, base = mini_csv_dataset
        output_dir = base / "yolo_out"

        preparer = YoloDatasetPreparer(
            source_images_dir=images_dir,
            annotations_csv=csv_path,
            output_dir=output_dir,
            class_names=["liver"],
            train_ratio=0.75,
        )
        yaml_path = preparer.prepare()

        assert yaml_path.exists()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()
        assert (output_dir / "val" / "images").is_dir()
        assert (output_dir / "val" / "labels").is_dir()

        # Check that labels are in YOLO format
        all_labels = list((output_dir / "train" / "labels").glob("*.txt")) + \
                     list((output_dir / "val" / "labels").glob("*.txt"))
        assert len(all_labels) == 4

        for label_path in all_labels:
            content = label_path.read_text().strip()
            assert content  # Not empty
            parts = content.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert class_id == 0
            for val in parts[1:]:
                f = float(val)
                assert 0.0 <= f <= 1.0

    def test_data_yaml_content(
        self, mini_csv_dataset: tuple[Path, Path, Path]
    ) -> None:
        images_dir, csv_path, base = mini_csv_dataset
        output_dir = base / "yolo_out2"

        preparer = YoloDatasetPreparer(
            source_images_dir=images_dir,
            annotations_csv=csv_path,
            output_dir=output_dir,
            class_names=["liver"],
        )
        yaml_path = preparer.prepare()
        content = yaml_path.read_text()

        assert "nc: 1" in content
        assert "names:" in content
        assert "liver" in content
        assert "train:" in content
        assert "val:" in content

    def test_prepare_with_single_image_uses_image_for_both_splits(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images_dir / "solo.png")

        csv_path = tmp_path / "train.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "solo",
                    "x_min": "4",
                    "y_min": "5",
                    "x_max": "20",
                    "y_max": "21",
                    "class_id": "0",
                }
            )

        output_dir = tmp_path / "yolo_out"
        preparer = YoloDatasetPreparer(
            source_images_dir=images_dir,
            annotations_csv=csv_path,
            output_dir=output_dir,
            class_names=["liver"],
        )
        preparer.prepare()

        assert (output_dir / "train" / "images" / "solo.png").is_file()
        assert (output_dir / "val" / "images" / "solo.png").is_file()

    def test_prepare_clears_stale_outputs_between_runs(
        self, mini_csv_dataset: tuple[Path, Path, Path]
    ) -> None:
        images_dir, csv_path, base = mini_csv_dataset
        output_dir = base / "yolo_out_reset"

        preparer = YoloDatasetPreparer(
            source_images_dir=images_dir,
            annotations_csv=csv_path,
            output_dir=output_dir,
            class_names=["liver"],
        )
        preparer.prepare()

        stale_label = output_dir / "train" / "labels" / "stale.txt"
        stale_label.write_text("junk\n", encoding="utf-8")

        preparer.prepare()

        assert not stale_label.exists()

    def test_prepare_skips_invalid_rows_and_missing_images(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(images_dir / "keep.png")

        csv_path = tmp_path / "train.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "keep",
                    "x_min": "2",
                    "y_min": "3",
                    "x_max": "20",
                    "y_max": "21",
                    "class_id": "0",
                }
            )
            writer.writerow(
                {
                    "image_id": "keep",
                    "x_min": "8",
                    "y_min": "9",
                    "x_max": "4",
                    "y_max": "10",
                    "class_id": "0",
                }
            )
            writer.writerow(
                {
                    "image_id": "missing",
                    "x_min": "2",
                    "y_min": "3",
                    "x_max": "20",
                    "y_max": "21",
                    "class_id": "0",
                }
            )
            writer.writerow(
                {
                    "image_id": "keep",
                    "x_min": "2",
                    "y_min": "3",
                    "x_max": "20",
                    "y_max": "21",
                    "class_id": "5",
                }
            )

        output_dir = tmp_path / "yolo_out_filtered"
        preparer = YoloDatasetPreparer(
            source_images_dir=images_dir,
            annotations_csv=csv_path,
            output_dir=output_dir,
            class_names=["liver"],
        )
        preparer.prepare()

        label_files = list((output_dir / "train" / "labels").glob("*.txt")) + list(
            (output_dir / "val" / "labels").glob("*.txt")
        )
        assert len(label_files) >= 1
        contents = "\n".join(path.read_text(encoding="utf-8") for path in label_files)
        assert "5 " not in contents


# ---------------------------------------------------------------------------
# YoloTrainingConfig tests
# ---------------------------------------------------------------------------

class TestYoloTrainingConfig:
    def test_default_config(self, tmp_path: Path) -> None:
        config = YoloTrainingConfig(dataset_yaml=tmp_path / "data.yaml")
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.image_size == 640

    def test_invalid_epochs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            YoloTrainingConfig(dataset_yaml=tmp_path / "data.yaml", epochs=0)

    def test_invalid_batch_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            YoloTrainingConfig(dataset_yaml=tmp_path / "data.yaml", batch_size=0)

    def test_invalid_learning_rate(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            YoloTrainingConfig(dataset_yaml=tmp_path / "data.yaml", learning_rate=0)

    def test_invalid_image_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="image_size must be >= 32"):
            YoloTrainingConfig(dataset_yaml=tmp_path / "data.yaml", image_size=16)


# ---------------------------------------------------------------------------
# yolo_utils tests
# ---------------------------------------------------------------------------

class TestYoloUtils:
    def test_mask_to_xyxy(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 255
        box = mask_to_xyxy(mask)
        assert box is not None
        assert box.x1 == 30.0
        assert box.y1 == 20.0
        assert box.x2 == 79.0
        assert box.y2 == 59.0

    def test_mask_to_xyxy_empty(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert mask_to_xyxy(mask) is None

    def test_xyxy_to_yolo_label(self) -> None:
        bbox = YoloXyxyBox(x1=10.0, y1=20.0, x2=90.0, y2=80.0)
        label = xyxy_to_yolo_label(
            bbox=bbox, class_id=0, class_name="liver",
            image_width=100, image_height=100,
        )
        assert label.class_id == 0
        assert label.class_name == "liver"
        assert 0.0 <= label.x_center <= 1.0
        assert 0.0 <= label.y_center <= 1.0
        assert 0.0 < label.width <= 1.0
        assert 0.0 < label.height <= 1.0

    def test_format_and_parse_labels(self) -> None:
        labels = [
            YoloLabel(class_id=0, class_name="liver",
                          x_center=0.5, y_center=0.5, width=0.8, height=0.6),
        ]
        txt = format_yolo_labels(labels)
        assert "0 0.500000 0.500000 0.800000 0.600000" in txt

        parsed = parse_yolo_txt_labels(txt, class_names=["liver"])
        assert len(parsed) == 1
        assert parsed[0].class_id == 0
        assert abs(parsed[0].x_center - 0.5) < 1e-4

    def test_format_empty_labels(self) -> None:
        assert format_yolo_labels([]) == ""

    def test_parse_invalid_label(self) -> None:
        with pytest.raises(ValueError, match="expected 5 columns"):
            parse_yolo_txt_labels("0 0.5 0.5", class_names=["liver"])

    def test_parse_labels_without_class_names_accepts_multiclass_ids(self) -> None:
        parsed = parse_yolo_txt_labels("2 0.5 0.5 0.2 0.2\n")
        assert len(parsed) == 1
        assert parsed[0].class_id == 2
        assert parsed[0].class_name is None

    def test_xyxy_to_yolo_label_clips_to_image_bounds(self) -> None:
        bbox = YoloXyxyBox(x1=90.0, y1=95.0, x2=140.0, y2=160.0)
        label = xyxy_to_yolo_label(
            bbox=bbox,
            class_id=0,
            class_name="liver",
            image_width=100,
            image_height=100,
        )
        assert 0.0 <= label.x_center <= 1.0
        assert 0.0 <= label.y_center <= 1.0
        assert 0.0 < label.width <= 1.0
        assert 0.0 < label.height <= 1.0

    def test_xyxy_to_yolo_label_rejects_inverted_boxes(self) -> None:
        bbox = YoloXyxyBox(x1=10.0, y1=20.0, x2=5.0, y2=25.0)
        with pytest.raises(ValueError, match="x2 >= x1"):
            xyxy_to_yolo_label(
                bbox=bbox,
                class_id=0,
                class_name="liver",
                image_width=100,
                image_height=100,
            )



