"""Tests for reusable notebook-oriented workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests._transcranial_fixtures import create_transcranial_fixture
from ultrasound.workflows import (
    run_dataset_healthcheck,
    run_masked_proximal_decomposition,
    run_mini_training_pipeline,
    run_model_metric_smoke,
    run_ndt_ascan_analysis,
    run_phase_retrieval_transcranial,
    run_phase_retrieval_ultrasound,
    run_preprocessing_workbench,
    tune_phase_retrieval_transcranial,
)


def _sample_image() -> np.ndarray:
    x = np.linspace(0, 255, 64, dtype=np.uint8)
    grid_x, grid_y = np.meshgrid(x, x)
    image = np.stack([grid_x, grid_y, ((grid_x.astype(np.uint16) + grid_y.astype(np.uint16)) // 2).astype(np.uint8)], axis=2)
    return image.astype(np.uint8)


def _sample_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:46, 20:44] = 255
    return mask


def test_run_preprocessing_workbench_returns_expected_artifacts() -> None:
    result = run_preprocessing_workbench(_sample_image(), n_iter=8)

    assert result.gray.shape == (64, 64)
    assert set(result.processed) == {"Lee", "Frost", "Median", "CLAHE", "ADMM-TV"}
    assert set(result.quality) == {"lee", "frost", "tv"}
    assert set(result.speckle_cv) == {"original", "lee", "frost", "tv"}
    assert result.convergence["primal_residuals"]
    for image in result.processed.values():
        assert image.shape == result.gray.shape


def test_run_model_metric_smoke_returns_numeric_summaries() -> None:
    result = run_model_metric_smoke(_sample_image(), _sample_mask(), seed=7)

    assert result.shapes["unet_logits"] == [2, 1, 128, 128]
    assert result.shapes["cnn_logits"][0] == 2
    assert set(result.losses) == {"dice_loss", "combined_loss", "focal_loss"}
    assert result.segmentation_metrics["iou"] >= 0.0
    assert result.confusion_matrix.shape == (2, 2)
    assert "accuracy" in result.classification_metrics


def test_run_dataset_healthcheck_builds_overlays_and_status() -> None:
    ndt_rows = [{"sample": "demo.npz", "n_defects": 2}]

    result = run_dataset_healthcheck(
        {"benign": 3, "malignant": 2, "normal": 0},
        {
            "benign": (_sample_image(), _sample_mask()),
            "malignant": (_sample_image(), _sample_mask()),
        },
        ndt_rows,
        seed=9,
    )

    assert result.health_status["busi_ready"] is True
    assert result.health_status["ndt_ready"] is True
    assert result.health_status["overall_ready"] is True
    assert result.report["seed"] == 9
    assert set(result.overlays) == {"benign", "malignant"}
    for overlay in result.overlays.values():
        assert overlay.shape == _sample_image().shape


def test_run_mini_training_pipeline_executes_two_step_smoke_loop() -> None:
    torch.manual_seed(5)
    images = torch.rand(4, 3, 96, 96)
    masks = (torch.rand(4, 1, 96, 96) > 0.5).float()
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, masks, labels), batch_size=2, shuffle=False)

    result = run_mini_training_pipeline(loader, seed=5, steps=2, device="cpu")

    assert len(result.segmentation_losses) == 2
    assert len(result.classification_losses) == 2
    assert result.status["overall_pass"] is True


def test_run_ndt_ascan_analysis_detects_echoes() -> None:
    n = 256
    fs_hz = 20e6
    time_s = np.arange(n, dtype=np.float64) / fs_hz
    rf = np.zeros(n, dtype=np.float64)
    rf[20] = 1.0
    rf[80] = 0.8
    rf[140] = 0.4

    result = run_ndt_ascan_analysis(
        rf,
        time_s,
        fs_hz=fs_hz,
        fc_hz=5e6,
        c_mps=5900.0,
        nominal_thickness_m=0.01,
    )

    assert result.peak_indices.size >= 2
    assert len(result.peak_times_us) >= 2
    assert result.status["overall_pass"] is True


def test_run_phase_retrieval_ultrasound_recovers_waveform_from_stft_magnitude() -> None:
    samples = np.arange(256, dtype=np.float64)
    centered = samples - 104.0
    rf = np.exp(-0.5 * (centered / 15.0) ** 2) * (
        np.cos(2.0 * np.pi * 0.10 * centered) + 0.12 * np.cos(2.0 * np.pi * 0.18 * centered)
    )

    result = run_phase_retrieval_ultrasound(rf, seed=4, n_fft=64, hop_length=8, n_iter=100)

    assert result.status["error_reduced"] is True
    assert result.report["final_relative_error"] < 0.25
    assert result.report["signal_correlation"] > 0.95
    assert result.measured_spectrogram.shape == result.reconstructed_spectrogram.shape
    assert result.residual_curve


def test_run_phase_retrieval_transcranial_uses_real_hydrophone_fixture(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    result = run_phase_retrieval_transcranial(
        root_dir=str(tmp_path / "phase_retrieval"),
        case_name="Parietal_free_field_0_XY",
        window_length=256,
        n_fft=80,
        hop_length=8,
        n_iter=120,
        seed=7,
    )

    assert result.signal_metadata is not None
    assert result.signal_metadata["case_name"] == "Parietal_free_field_0_XY"
    assert result.report["final_relative_error"] < 0.25
    assert result.status["overall_pass"] is True
    assert result.scan_energy_map is not None


def test_tune_phase_retrieval_transcranial_returns_best_config(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    tuning = tune_phase_retrieval_transcranial(
        root_dir=str(tmp_path / "phase_retrieval"),
        cases=["Frontal_40_XY", "Parietal_free_field_0_XY"],
        window_lengths=(256,),
        n_fft_grid=(64, 80),
        hop_length_grid=(8,),
        iteration_grid=(80,),
        seed=11,
    )

    assert tuning.best_config["window_length"] == 256
    assert tuning.best_config["n_fft"] in {64, 80}
    assert tuning.ranked_results
    assert tuning.cases == ["Frontal_40_XY", "Parietal_free_field_0_XY"]


def test_run_masked_proximal_decomposition_beats_zero_fill_baseline() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float64)
    signal = np.sin(x)
    signal[80] += 0.8
    signal[160] -= 0.6

    result = run_masked_proximal_decomposition(signal, seed=3, n_iter=40)

    assert result.status["improves_over_baseline"] is True
    assert result.status["objective_decreases"] is True
    assert len(result.objective_history) == 40
