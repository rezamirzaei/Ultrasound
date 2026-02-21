"""API integration tests for REST endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ultrasound.api.app import create_app
from ultrasound.api.config import AppConfig
from ultrasound.data import create_sample_data


def _create_ndt_fixture(ndt_dir: Path) -> None:
    """Create one deterministic NDT sample for isolated API tests."""
    ndt_dir.mkdir(parents=True, exist_ok=True)

    n = 512
    fs_hz = 50e6
    time = np.arange(n, dtype=np.float64) / fs_hz
    rf = np.sin(2.0 * np.pi * 5e6 * time) * np.exp(-time * 8e5)

    np.savez(
        ndt_dir / "synthetic_sample.npz",
        rf=rf,
        time=time,
        fs=fs_hz,
        fc=5e6,
        c=5900.0,
        thickness=0.01,
        description="Synthetic NDT sample for API tests",
        defects=np.array([[0.005, 0.4], [0.007, np.nan]], dtype=np.float64),
    )


def _create_ui_fixture(ui_dir: Path) -> None:
    """Create a minimal UI index file for static serving tests."""
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text(
        "<!doctype html><html><head><title>inPhase Ultrasound Platform</title></head>"
        "<body><h1>inPhase Ultrasound Platform</h1></body></html>",
        encoding="utf-8",
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create an isolated app instance with synthetic data and UI fixtures."""
    data_dir = tmp_path / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = tmp_path / "ui"
    artifacts_dir = tmp_path / "artifacts"

    create_sample_data(str(busi_dir), num_samples=2)
    _create_ndt_fixture(ndt_dir)
    _create_ui_fixture(ui_dir)

    config = AppConfig(
        project_root=tmp_path,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
    )
    app = create_app(config=config)
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/index.html"


def test_ui_index_served(client: TestClient) -> None:
    response = client.get("/ui/index.html")

    assert response.status_code == 200
    assert "inPhase Ultrasound Platform" in response.text


def test_dashboard_summary_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "busi_counts" in payload
    assert "busi_total" in payload
    assert payload["busi_total"] >= 0


def test_dashboard_readiness_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "warning"}
    assert isinstance(payload["issues"], list)
    assert payload["ndt_samples"] >= 0
    assert isinstance(payload["busi_available_classes"], list)
    assert isinstance(payload["busi_missing_classes"], list)


def test_ndt_sample_listing_and_detail(client: TestClient) -> None:
    list_response = client.get("/api/v1/datasets/ndt/samples")

    assert list_response.status_code == 200
    samples = list_response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0

    detail_response = client.get(f"/api/v1/datasets/ndt/samples/{samples[0]['name']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["name"] == samples[0]["name"]
    assert detail["n_points"] > 0

    # Verify defects parsed correctly from 2D float array
    assert detail["n_defects"] >= 1
    for defect in detail["defects"]:
        # depth_m should be a valid float or None
        assert defect["depth_m"] is None or isinstance(defect["depth_m"], float)
        # amplitude should be a valid float or None (nan should become None)
        assert defect["amplitude"] is None or isinstance(defect["amplitude"], float)
        assert defect["source"] in {"metadata", "signal", "fused"}
        assert 0.0 <= defect["confidence"] <= 1.0
        if defect["time_us"] is not None:
            assert defect["time_us"] >= 0.0

    signal_response = client.get(
        f"/api/v1/datasets/ndt/samples/{samples[0]['name']}/signal",
        params={"max_points": 256},
    )
    assert signal_response.status_code == 200
    signal = signal_response.json()
    assert signal["sample_name"] == samples[0]["name"]
    assert signal["n_original_points"] >= signal["n_sampled_points"] > 0
    assert len(signal["time_us"]) == signal["n_sampled_points"]
    assert len(signal["rf"]) == signal["n_sampled_points"]
    assert "stats" in signal

    # Verify defect markers are valid — only defects with finite depth produce markers
    for marker in signal["defect_markers"]:
        assert marker["depth_mm"] >= 0
        assert marker["two_way_time_us"] >= 0
        assert marker["amplitude"] is None or isinstance(marker["amplitude"], float)
        assert marker["source"] in {"metadata", "signal", "fused"}
        assert 0.0 <= marker["confidence"] <= 1.0


def test_ndt_defect_parsing_with_tuple_format(tmp_path: Path) -> None:
    """Test that defects saved as list-of-tuples (download_ascan_data format) parse correctly."""
    data_dir = tmp_path / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = tmp_path / "ui"

    create_sample_data(str(busi_dir), num_samples=1)
    ndt_dir.mkdir(parents=True, exist_ok=True)
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<html></html>")

    # Save defects as list-of-tuples — exactly how download_ascan_data.py does it
    n = 256
    fs = 50e6
    np.savez(
        ndt_dir / "tuple_defects.npz",
        rf=np.random.randn(n),
        time=np.arange(n, dtype=np.float64) / fs,
        fs=fs,
        fc=5e6,
        c=5900.0,
        thickness=0.02,
        description="Test with tuple defects",
        defects=[(0.008, 0.4), (0.011, 0.2)],
    )

    config = AppConfig(
        project_root=tmp_path,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=tmp_path / "artifacts",
    )
    app = create_app(config=config)
    test_client = TestClient(app)

    detail = test_client.get("/api/v1/datasets/ndt/samples/tuple_defects.npz").json()
    assert detail["n_defects"] == 2
    assert detail["defects"][0]["depth_m"] == pytest.approx(0.008)
    assert detail["defects"][0]["amplitude"] == pytest.approx(0.4)
    assert detail["defects"][1]["depth_m"] == pytest.approx(0.011)
    assert detail["defects"][1]["amplitude"] == pytest.approx(0.2)

    signal = test_client.get(
        "/api/v1/datasets/ndt/samples/tuple_defects.npz/signal",
        params={"max_points": 256},
    ).json()
    assert len(signal["defect_markers"]) == 2
    assert signal["defect_markers"][0]["depth_mm"] == pytest.approx(8.0)
    assert signal["defect_markers"][1]["depth_mm"] == pytest.approx(11.0)


def test_ndt_signal_detection_when_metadata_is_empty(tmp_path: Path) -> None:
    """Defects should still be detected from waveform when metadata has no defect labels."""
    data_dir = tmp_path / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = tmp_path / "ui"

    create_sample_data(str(busi_dir), num_samples=1)
    ndt_dir.mkdir(parents=True, exist_ok=True)
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<html></html>")

    fs = 50e6
    fc = 5e6
    c = 5900.0
    thickness_m = 0.012
    n = 1200
    time = np.arange(n, dtype=np.float64) / fs

    pulse_duration_s = 0.5e-6
    pulse_t = np.arange(0.0, pulse_duration_s, 1.0 / fs)
    pulse = np.exp(
        -((pulse_t - pulse_duration_s / 2.0) ** 2) / (2.0 * (pulse_duration_s / 6.0) ** 2)
    )
    pulse *= np.sin(2.0 * np.pi * fc * pulse_t)

    rf = np.zeros(n, dtype=np.float64)
    fw_idx = int(0.5e-6 * fs)
    defect_depth_m = 0.006
    defect_idx = int((2.0 * defect_depth_m / c) * fs)
    bw_idx = int((2.0 * thickness_m / c) * fs)

    rf[fw_idx : fw_idx + pulse.size] += pulse
    rf[defect_idx : defect_idx + pulse.size] += 0.38 * pulse
    rf[bw_idx : bw_idx + pulse.size] += 0.8 * pulse

    rng = np.random.default_rng(123)
    rf += 0.01 * rng.standard_normal(n)

    np.savez(
        ndt_dir / "signal_only_defect.npz",
        rf=rf,
        time=time,
        fs=fs,
        fc=fc,
        c=c,
        thickness=thickness_m,
        description="Signal-only defect test sample",
        defects=np.array([], dtype=np.float64),
    )

    config = AppConfig(
        project_root=tmp_path,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=tmp_path / "artifacts",
    )
    app = create_app(config=config)
    test_client = TestClient(app)

    detail = test_client.get("/api/v1/datasets/ndt/samples/signal_only_defect.npz").json()
    assert detail["n_defects"] >= 1
    signal_like = [item for item in detail["defects"] if item["source"] in {"signal", "fused"}]
    assert signal_like
    assert any(0.004 <= float(item["depth_m"]) <= 0.008 for item in signal_like)
    assert all(0.0 <= float(item["confidence"]) <= 1.0 for item in signal_like)

    signal = test_client.get(
        "/api/v1/datasets/ndt/samples/signal_only_defect.npz/signal",
        params={"max_points": 512},
    ).json()
    assert signal["defect_markers"]
    assert any(marker["source"] == "signal" for marker in signal["defect_markers"])


def test_busi_sample_preview_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/datasets/busi/samples/benign/0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["class_name"] == "benign"
    assert payload["total_samples"] > 0
    assert payload["image_data_url"].startswith("data:image/png;base64,")
    assert payload["mask_data_url"].startswith("data:image/png;base64,")
    assert payload["lesion_pixels"] >= 0
    assert 0.0 <= payload["lesion_ratio"] <= 1.0


def test_preprocessing_preview_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/v1/preprocessing/preview",
        json={
            "class_name": "benign",
            "sample_index": 0,
            "lambda_tv": 0.04,
            "rho": 1.0,
            "n_iter": 8,
            "clip_limit": 2.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation"]
    assert payload["original_image_data_url"].startswith("data:image/png;base64,")
    assert len(payload["methods"]) == 4
