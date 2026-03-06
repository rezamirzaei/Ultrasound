"""Smoke tests for demo entrypoints and orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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


class _TinyModel(torch.nn.Module):
    def __init__(self, output_channels: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (x.shape[0], self.output_channels, x.shape[2], x.shape[3]),
            dtype=x.dtype,
        )


def test_demo_segmentation_runs_with_lightweight_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo, "UNet", lambda **_kwargs: _TinyModel(output_channels=1))
    monkeypatch.setattr(demo, "AttentionUNet", lambda **_kwargs: _TinyModel(output_channels=1))
    monkeypatch.setattr(demo, "dice_loss", lambda *_args, **_kwargs: torch.tensor(0.25))
    monkeypatch.setattr(demo, "combined_loss", lambda *_args, **_kwargs: torch.tensor(0.5))

    demo.demo_segmentation(tmp_path / "nested" / "outputs")


def test_demo_classification_falls_back_when_pretrained_weights_are_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(demo, "UltrasoundClassifier", lambda **_kwargs: _TinyModel(output_channels=2))

    def _make_resnet(**kwargs: object) -> _TinyModel:
        pretrained = bool(kwargs["pretrained"])
        calls.append(pretrained)
        if pretrained:
            raise RuntimeError("offline")
        return _TinyModel(output_channels=2)

    monkeypatch.setattr(demo, "ResNetClassifier", _make_resnet)

    demo.demo_classification(tmp_path / "nested" / "outputs")

    assert calls == [True, False]


def test_demo_admm_optimization_creates_analysis_figure(monkeypatch, tmp_path: Path) -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    monkeypatch.setattr(
        demo,
        "admm_tv_denoising",
        lambda gray, **_kwargs: (
            gray,
            {"primal_residuals": [1.0, 0.5], "dual_residuals": [0.75, 0.25]},
        ),
    )

    demo.demo_admm_optimization(image, tmp_path / "results")

    assert (tmp_path / "results" / "admm_analysis.png").exists()


def test_demo_full_pipeline_generates_sample_preview(monkeypatch, tmp_path: Path) -> None:
    create_calls: list[tuple[str, int]] = []

    class _DatasetStub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.sample = (
                torch.zeros((3, 8, 8), dtype=torch.float32),
                torch.zeros((1, 8, 8), dtype=torch.float32),
                1,
            )

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return self.sample

    monkeypatch.setattr(demo, "BUSIDataset", _DatasetStub)
    monkeypatch.setattr(
        demo,
        "create_sample_data",
        lambda root, num_samples: create_calls.append((str(root), int(num_samples))),
    )

    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    demo.demo_full_pipeline(data_dir, output_dir)

    assert create_calls == [(str(data_dir), 5)]
    assert (output_dir / "sample_data.png").exists()


def test_demo_full_pipeline_skips_preview_when_dataset_is_empty(monkeypatch, tmp_path: Path) -> None:
    class _EmptyDatasetStub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(demo, "BUSIDataset", _EmptyDatasetStub)
    monkeypatch.setattr(demo, "create_sample_data", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    demo.demo_full_pipeline(data_dir, output_dir)

    assert not (output_dir / "sample_data.png").exists()
