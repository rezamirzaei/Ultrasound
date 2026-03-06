"""Smoke tests for demo entrypoints and orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ultrasound.demo as demo


def test_demo_preprocessing_writes_expected_outputs(tmp_path: Path) -> None:
    base = np.tile(np.linspace(0, 255, 24, dtype=np.uint8), (24, 1))
    image = np.stack([base, base, base], axis=-1)

    processed = demo.demo_preprocessing(image, tmp_path)

    assert set(processed) == {"Lee Filter", "Frost Filter", "CLAHE", "ADMM-TV"}
    assert (tmp_path / "preprocessing_comparison.png").exists()
    assert (tmp_path / "speckle_analysis.png").exists()


def test_main_runs_selected_demos_once_and_generates_synthetic_input(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        demo,
        "_generate_synthetic_ultrasound",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        demo,
        "demo_preprocessing",
        lambda image, out: calls.append(("preprocessing", (image.shape, out))),
    )
    monkeypatch.setattr(
        demo,
        "demo_admm_optimization",
        lambda image, out: calls.append(("admm", (image.shape, out))),
    )
    monkeypatch.setattr(demo, "demo_segmentation", lambda out: calls.append(("segmentation", out)))
    monkeypatch.setattr(
        demo,
        "demo_classification",
        lambda out: calls.append(("classification", out)),
    )
    monkeypatch.setattr(
        demo,
        "demo_full_pipeline",
        lambda data, out: calls.append(("pipeline", (data, out))),
    )

    demo.main(
        demos=["preprocessing", "admm", "segmentation", "classification", "pipeline"],
        output_dir=output_dir,
        data_dir=data_dir,
    )

    assert ("segmentation", output_dir) in calls
    assert ("classification", output_dir) in calls
    assert ("pipeline", (data_dir, output_dir)) in calls
    assert calls.count(("segmentation", output_dir)) == 1
    assert output_dir.joinpath("synthetic_ultrasound.png").exists()


def test_parse_args_supports_custom_paths() -> None:
    args = demo._parse_args(["--demo", "preprocessing", "admm", "--output-dir", "out", "--data-dir", "busi"])

    assert args.demo == ["preprocessing", "admm"]
    assert args.output_dir == Path("out")
    assert args.data_dir == Path("busi")
