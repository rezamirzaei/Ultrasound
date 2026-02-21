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


def _auth_headers(
    client: TestClient,
    username: str = "viewer",
    password: str = "viewer123",
) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


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


def test_auth_login_and_me_endpoint(client: TestClient) -> None:
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "viewer", "password": "viewer123"},
    )
    assert login_response.status_code == 200
    payload = login_response.json()
    assert payload["token_type"] == "Bearer"
    assert payload["role"] == "viewer"
    assert payload["access_token"]

    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert me_response.status_code == 200
    me_payload = me_response.json()
    assert me_payload["username"] == "viewer"
    assert me_payload["role"] == "viewer"


def test_auth_logout_revokes_token(client: TestClient) -> None:
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "viewer", "password": "viewer123"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    logout_response = client.post("/api/v1/auth/logout", headers=headers)
    assert logout_response.status_code == 200
    payload = logout_response.json()
    assert payload["success"] is True
    assert payload["revoked_token"] is True

    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 401
    assert "Token revoked" in me_response.json()["detail"]


def test_protected_endpoint_requires_authentication(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 401
    payload = response.json()
    assert "detail" in payload
    assert "request_id" in payload


def test_dashboard_summary_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/summary", headers=_auth_headers(client))

    assert response.status_code == 200
    payload = response.json()
    assert "busi_counts" in payload
    assert "busi_total" in payload
    assert payload["busi_total"] >= 0


def test_dashboard_readiness_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/readiness", headers=_auth_headers(client))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "warning"}
    assert isinstance(payload["issues"], list)
    assert payload["ndt_samples"] >= 0
    assert isinstance(payload["busi_available_classes"], list)
    assert isinstance(payload["busi_missing_classes"], list)


def test_ndt_sample_listing_and_detail(client: TestClient) -> None:
    headers = _auth_headers(client)
    list_response = client.get("/api/v1/datasets/ndt/samples", headers=headers)

    assert list_response.status_code == 200
    samples = list_response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0

    detail_response = client.get(
        f"/api/v1/datasets/ndt/samples/{samples[0]['name']}", headers=headers
    )
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
        headers=headers,
    )
    assert signal_response.status_code == 200
    signal = signal_response.json()
    assert signal["sample_name"] == samples[0]["name"]
    assert signal["n_original_points"] >= signal["n_sampled_points"] > 0
    assert len(signal["time_us"]) == signal["n_sampled_points"]
    assert len(signal["rf"]) == signal["n_sampled_points"]
    assert "stats" in signal
    assert signal["total_peaks"] >= 0
    assert isinstance(signal["thinning_flag"], bool)
    assert isinstance(signal["wall_markers"], list)
    assert signal["thickness_method"] in {
        "time_of_flight",
        "absolute_backwall",
        "insufficient_data",
    }
    if signal["estimated_thickness_mm"] is not None:
        assert signal["estimated_thickness_mm"] >= 0.0
    if signal["thickness_std_mm"] is not None:
        assert signal["thickness_std_mm"] >= 0.0
    if signal["thickness_ci95_lower_mm"] is not None:
        assert signal["thickness_ci95_lower_mm"] >= 0.0
    if signal["thickness_ci95_upper_mm"] is not None:
        assert signal["thickness_ci95_upper_mm"] >= signal["thickness_ci95_lower_mm"]
    if signal["thickness_confidence"] is not None:
        assert 0.0 <= signal["thickness_confidence"] <= 1.0
    if signal["nominal_thickness_mm"] is not None:
        assert signal["nominal_thickness_mm"] >= 0.0

    # Verify defect markers are valid — only defects with finite depth produce markers
    for marker in signal["defect_markers"]:
        assert marker["depth_mm"] >= 0
        assert marker["two_way_time_us"] >= 0
        assert marker["amplitude"] is None or isinstance(marker["amplitude"], float)
        assert marker["source"] in {"metadata", "signal", "fused"}
        assert 0.0 <= marker["confidence"] <= 1.0

    for wall in signal["wall_markers"]:
        assert wall["label"] in {"front_wall", "back_wall"}
        assert wall["two_way_time_us"] >= 0
        assert wall["amplitude"] is None or isinstance(wall["amplitude"], float)
        if wall["confidence"] is not None:
            assert 0.0 <= wall["confidence"] <= 1.0
        if wall["time_std_us"] is not None:
            assert wall["time_std_us"] >= 0.0
        if wall["depth_mm"] is not None:
            assert wall["depth_mm"] >= 0


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
    n = 1024
    fs = 50e6
    fc = 5e6
    c = 5900.0
    thickness_m = 0.02
    time = np.arange(n, dtype=np.float64) / fs
    pulse_duration_s = 0.5e-6
    pulse_t = np.arange(0.0, pulse_duration_s, 1.0 / fs)
    pulse = np.exp(
        -((pulse_t - pulse_duration_s / 2.0) ** 2) / (2.0 * (pulse_duration_s / 6.0) ** 2)
    )
    pulse *= np.sin(2.0 * np.pi * fc * pulse_t)

    rf = np.zeros(n, dtype=np.float64)
    fw_idx = int(0.5e-6 * fs)
    d1_idx = int((2.0 * 0.008 / c) * fs)
    d2_idx = int((2.0 * 0.011 / c) * fs)
    bw_idx = int((2.0 * thickness_m / c) * fs)
    rf[fw_idx : fw_idx + pulse.size] += pulse
    rf[d1_idx : d1_idx + pulse.size] += 0.4 * pulse
    rf[d2_idx : d2_idx + pulse.size] += 0.2 * pulse
    rf[bw_idx : bw_idx + pulse.size] += 0.8 * pulse

    rng = np.random.default_rng(5)
    rf += 0.01 * rng.standard_normal(n)

    np.savez(
        ndt_dir / "tuple_defects.npz",
        rf=rf,
        time=time,
        fs=fs,
        fc=fc,
        c=c,
        thickness=thickness_m,
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

    headers = _auth_headers(test_client)
    detail = test_client.get(
        "/api/v1/datasets/ndt/samples/tuple_defects.npz", headers=headers
    ).json()
    assert detail["n_defects"] == 2
    depths = sorted(
        float(item["depth_m"]) for item in detail["defects"] if item["depth_m"] is not None
    )
    assert len(depths) == 2
    assert abs(depths[0] - 0.008) < 0.0015
    assert abs(depths[1] - 0.011) < 0.0015

    signal = test_client.get(
        "/api/v1/datasets/ndt/samples/tuple_defects.npz/signal",
        params={"max_points": 256},
        headers=headers,
    ).json()
    assert signal["total_peaks"] >= 2
    assert signal["thinning_flag"] is False
    assert len(signal["wall_markers"]) == 2
    assert len(signal["defect_markers"]) == 2
    marker_depths = sorted(float(item["depth_mm"]) for item in signal["defect_markers"])
    assert abs(marker_depths[0] - 8.0) < 1.5
    assert abs(marker_depths[1] - 11.0) < 1.5


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

    headers = _auth_headers(test_client)
    detail = test_client.get(
        "/api/v1/datasets/ndt/samples/signal_only_defect.npz", headers=headers
    ).json()
    assert detail["n_defects"] >= 1
    signal_like = [item for item in detail["defects"] if item["source"] in {"signal", "fused"}]
    assert signal_like
    assert any(0.004 <= float(item["depth_m"]) <= 0.008 for item in signal_like)
    assert all(0.0 <= float(item["confidence"]) <= 1.0 for item in signal_like)

    signal = test_client.get(
        "/api/v1/datasets/ndt/samples/signal_only_defect.npz/signal",
        params={"max_points": 512},
        headers=headers,
    ).json()
    assert signal["total_peaks"] >= 3
    assert signal["thinning_flag"] is False
    assert len(signal["wall_markers"]) == 2
    assert signal["estimated_thickness_mm"] is not None
    assert signal["defect_markers"]
    assert any(marker["source"] == "signal" for marker in signal["defect_markers"])


def test_ndt_thinning_flag_and_thickness_estimation(tmp_path: Path) -> None:
    """Corrosion/thinning sample should expose wall markers and thinning metadata."""
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
    thickness_m = 0.008
    n = 1000
    time = np.arange(n, dtype=np.float64) / fs

    pulse_duration_s = 0.5e-6
    pulse_t = np.arange(0.0, pulse_duration_s, 1.0 / fs)
    pulse = np.exp(
        -((pulse_t - pulse_duration_s / 2.0) ** 2) / (2.0 * (pulse_duration_s / 6.0) ** 2)
    )
    pulse *= np.sin(2.0 * np.pi * fc * pulse_t)

    rf = np.zeros(n, dtype=np.float64)
    fw_idx = int(0.5e-6 * fs)
    bw_idx = int((2.0 * thickness_m / c) * fs)
    rf[fw_idx : fw_idx + pulse.size] += pulse
    rf[bw_idx : bw_idx + pulse.size] += 0.8 * pulse

    rng = np.random.default_rng(77)
    rf += 0.01 * rng.standard_normal(n)

    np.savez(
        ndt_dir / "corrosion_like.npz",
        rf=rf,
        time=time,
        fs=fs,
        fc=fc,
        c=c,
        thickness=thickness_m,
        description="Corroded plate (original 10mm)",
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

    headers = _auth_headers(test_client)
    signal = test_client.get(
        "/api/v1/datasets/ndt/samples/corrosion_like.npz/signal",
        params={"max_points": 512},
        headers=headers,
    ).json()

    assert signal["total_peaks"] >= 2
    assert len(signal["wall_markers"]) == 2
    assert signal["estimated_thickness_mm"] is not None
    assert signal["thinning_flag"] is True


def test_busi_sample_preview_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/datasets/busi/samples/benign/0", headers=_auth_headers(client))

    assert response.status_code == 200
    payload = response.json()
    assert payload["class_name"] == "benign"
    assert payload["total_samples"] > 0
    assert payload["image_data_url"].startswith("data:image/png;base64,")
    assert payload["mask_data_url"].startswith("data:image/png;base64,")
    assert payload["lesion_pixels"] >= 0
    assert 0.0 <= payload["lesion_ratio"] <= 1.0


def test_busi_training_endpoints(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")

    latest_before = client.get("/api/v1/datasets/busi/training/latest", headers=viewer_headers)
    assert latest_before.status_code == 200
    latest_before_payload = latest_before.json()
    assert latest_before_payload["storage"] == "sql"
    assert latest_before_payload["run_id"] is None
    assert latest_before_payload["curve"] == []

    forbidden = client.post(
        "/api/v1/datasets/busi/training/run",
        json={
            "include_normal": False,
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=viewer_headers,
    )
    assert forbidden.status_code == 403

    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")
    run_response = client.post(
        "/api/v1/datasets/busi/training/run",
        json={
            "include_normal": False,
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=analyst_headers,
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["run_id"] is not None
    assert run_payload["storage"] == "sql"
    assert run_payload["train_samples"] > 0
    assert run_payload["test_samples"] > 0
    assert 0.0 <= run_payload["train_accuracy"] <= 1.0
    assert 0.0 <= run_payload["test_accuracy"] <= 1.0
    assert len(run_payload["curve"]) == 4

    latest_after = client.get("/api/v1/datasets/busi/training/latest", headers=viewer_headers)
    assert latest_after.status_code == 200
    latest_after_payload = latest_after.json()
    assert latest_after_payload["run_id"] == run_payload["run_id"]
    assert latest_after_payload["curve"]


def test_preprocessing_preview_endpoint(client: TestClient) -> None:
    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")
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
        headers=analyst_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation"]
    assert payload["original_image_data_url"].startswith("data:image/png;base64,")
    assert len(payload["methods"]) == 4


def test_role_restriction_for_preprocessing(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")
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
        headers=viewer_headers,
    )
    assert response.status_code == 403
    assert "Role 'analyst'" in response.json()["detail"]


def test_ops_error_analytics_admin_only(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")
    admin_headers = _auth_headers(client, username="admin", password="admin123")

    forbidden = client.get("/api/v1/ops/errors/summary", headers=viewer_headers)
    assert forbidden.status_code == 403

    # Generate one controlled client-side error event to populate analytics.
    missing = client.get("/api/v1/datasets/ndt/samples/missing_sample.npz", headers=viewer_headers)
    assert missing.status_code == 404

    summary = client.get("/api/v1/ops/errors/summary", headers=admin_headers)
    assert summary.status_code == 200
    summary_payload = summary.json()
    assert summary_payload["total_error_count"] >= 1
    assert isinstance(summary_payload["by_status"], dict)

    recent = client.get("/api/v1/ops/errors/recent", headers=admin_headers)
    assert recent.status_code == 200
    recent_payload = recent.json()
    assert isinstance(recent_payload, list)

    resync_forbidden = client.post("/api/v1/ops/datasets/resync", headers=viewer_headers)
    assert resync_forbidden.status_code == 403

    resync = client.post("/api/v1/ops/datasets/resync", headers=admin_headers)
    assert resync.status_code == 200
    resync_payload = resync.json()
    assert resync_payload["busi_rows_synced"] >= 0
    assert resync_payload["ndt_rows_synced"] >= 0
    assert len(recent_payload) >= 1
    assert "request_id" in recent_payload[0]
