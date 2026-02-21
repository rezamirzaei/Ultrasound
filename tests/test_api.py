"""API integration tests for REST endpoints."""

from __future__ import annotations

import time
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


def _create_industrial_fixture(data_dir: Path) -> None:
    """Create minimal industrial dataset folders for ORM sync tests."""
    steel_train_crazing = (
        data_dir / "steel_defect" / "NEU Metal Surface Defects Data" / "train" / "Crazing"
    )
    steel_train_scale = (
        data_dir / "steel_defect" / "NEU Metal Surface Defects Data" / "train" / "Rolled_In_Scale"
    )
    steel_valid_crazing = (
        data_dir / "steel_defect" / "NEU Metal Surface Defects Data" / "valid" / "Crazing"
    )
    steel_train_crazing.mkdir(parents=True, exist_ok=True)
    steel_train_scale.mkdir(parents=True, exist_ok=True)
    steel_valid_crazing.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32), 120, dtype=np.uint8), mode="L").save(
        steel_train_crazing / "Cr_1.bmp"
    )
    Image.fromarray(np.full((32, 32), 80, dtype=np.uint8), mode="L").save(
        steel_train_scale / "RS_1.bmp"
    )
    Image.fromarray(np.full((32, 32), 135, dtype=np.uint8), mode="L").save(
        steel_valid_crazing / "Cr_2.bmp"
    )

    neu_train_crazing = data_dir / "neu_surface" / "NEU-DET" / "train" / "images" / "crazing"
    neu_train_inclusion = data_dir / "neu_surface" / "NEU-DET" / "train" / "images" / "inclusion"
    neu_valid_crazing = data_dir / "neu_surface" / "NEU-DET" / "validation" / "images" / "crazing"
    neu_valid_inclusion = (
        data_dir / "neu_surface" / "NEU-DET" / "validation" / "images" / "inclusion"
    )
    neu_train_annotations = data_dir / "neu_surface" / "NEU-DET" / "train" / "annotations"
    neu_valid_annotations = data_dir / "neu_surface" / "NEU-DET" / "validation" / "annotations"
    neu_train_crazing.mkdir(parents=True, exist_ok=True)
    neu_train_inclusion.mkdir(parents=True, exist_ok=True)
    neu_valid_crazing.mkdir(parents=True, exist_ok=True)
    neu_valid_inclusion.mkdir(parents=True, exist_ok=True)
    neu_train_annotations.mkdir(parents=True, exist_ok=True)
    neu_valid_annotations.mkdir(parents=True, exist_ok=True)

    def _write_neu_case(image_path: Path, annotation_path: Path, fill_value: int) -> None:
        Image.fromarray(np.full((24, 24), fill_value, dtype=np.uint8), mode="L").save(image_path)
        annotation_path.write_text(
            """
<annotation>
  <object>
    <name>defect</name>
    <bndbox>
      <xmin>4</xmin>
      <ymin>5</ymin>
      <xmax>16</xmax>
      <ymax>18</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
            encoding="utf-8",
        )

    _write_neu_case(
        neu_train_crazing / "crazing_1.jpg", neu_train_annotations / "crazing_1.xml", 180
    )
    _write_neu_case(
        neu_train_inclusion / "inclusion_1.jpg", neu_train_annotations / "inclusion_1.xml", 145
    )
    _write_neu_case(
        neu_valid_crazing / "crazing_2.jpg", neu_valid_annotations / "crazing_2.xml", 170
    )
    _write_neu_case(
        neu_valid_inclusion / "inclusion_2.jpg",
        neu_valid_annotations / "inclusion_2.xml",
        130,
    )

    casting_train_def = (
        data_dir / "casting_defect" / "casting_data" / "casting_data" / "train" / "def_front"
    )
    casting_train_ok = (
        data_dir / "casting_defect" / "casting_data" / "casting_data" / "train" / "ok_front"
    )
    casting_test_def = (
        data_dir / "casting_defect" / "casting_data" / "casting_data" / "test" / "def_front"
    )
    casting_train_def.mkdir(parents=True, exist_ok=True)
    casting_train_ok.mkdir(parents=True, exist_ok=True)
    casting_test_def.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((28, 28), 200, dtype=np.uint8), mode="L").save(
        casting_train_def / "cast_def_1.jpeg"
    )
    Image.fromarray(np.full((28, 28), 80, dtype=np.uint8), mode="L").save(
        casting_train_ok / "cast_ok_1.jpeg"
    )
    Image.fromarray(np.full((28, 28), 210, dtype=np.uint8), mode="L").save(
        casting_test_def / "cast_def_2.jpeg"
    )


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Create an isolated app instance with synthetic data and UI fixtures."""
    data_dir = tmp_path / "data"
    busi_dir = data_dir / "busi"
    ndt_dir = data_dir / "ascan_signals" / "ndt_samples"
    ui_dir = tmp_path / "ui"
    artifacts_dir = tmp_path / "artifacts"

    create_sample_data(str(busi_dir), num_samples=2)
    _create_ndt_fixture(ndt_dir)
    _create_ui_fixture(ui_dir)
    _create_industrial_fixture(data_dir)

    config = AppConfig(
        project_root=tmp_path,
        data_dir=data_dir,
        busi_dir=busi_dir,
        ndt_dir=ndt_dir,
        ui_dir=ui_dir,
        artifacts_dir=artifacts_dir,
    )
    app = create_app(config=config)
    with TestClient(app) as test_client:
        yield test_client


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


def test_industrial_summary_and_sample_preview(client: TestClient) -> None:
    headers = _auth_headers(client)

    summary_response = client.get("/api/v1/datasets/industrial/summary", headers=headers)
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["total_samples"] >= 3
    assert summary["totals_by_dataset"]["steel_defect"] >= 1
    assert summary["totals_by_dataset"]["neu_surface"] >= 1
    assert summary["totals_by_dataset"]["casting_defect"] >= 1
    assert len(summary["rows"]) >= 3

    steel_preview = client.get(
        "/api/v1/datasets/industrial/samples/steel_defect/train/crazing/0",
        headers=headers,
    )
    assert steel_preview.status_code == 200
    steel_payload = steel_preview.json()
    assert steel_payload["dataset_name"] == "steel_defect"
    assert steel_payload["split"] == "train"
    assert steel_payload["class_name"] == "crazing"
    assert steel_payload["total_samples"] >= 1
    assert steel_payload["image_data_url"].startswith("data:image/png;base64,")
    assert steel_payload["has_annotation"] is False

    neu_preview = client.get(
        "/api/v1/datasets/industrial/samples/neu_surface/train/crazing/0",
        headers=headers,
    )
    assert neu_preview.status_code == 200
    neu_payload = neu_preview.json()
    assert neu_payload["dataset_name"] == "neu_surface"
    assert neu_payload["has_annotation"] is True


def test_industrial_training_and_segmentation_endpoints(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")

    latest_before = client.get(
        "/api/v1/datasets/industrial/training/latest",
        params={"dataset_name": "neu_surface"},
        headers=viewer_headers,
    )
    assert latest_before.status_code == 200
    latest_before_payload = latest_before.json()
    assert latest_before_payload["dataset_name"] == "neu_surface"
    assert latest_before_payload["run_id"] is None
    assert latest_before_payload["curve"] == []
    assert latest_before_payload["task_type"] == "classification_single_label_with_bbox"
    assert latest_before_payload["classification_mode"] in {"binary", "multiclass"}
    assert latest_before_payload["label_source"] == "folder_name_plus_xml_bbox"
    assert latest_before_payload["segmentation_supported"] is True

    segmentation_preview = client.get(
        "/api/v1/datasets/industrial/segmentation/neu_surface/train/crazing/0",
        headers=viewer_headers,
    )
    assert segmentation_preview.status_code == 200
    seg_payload = segmentation_preview.json()
    assert seg_payload["dataset_name"] == "neu_surface"
    assert seg_payload["source"] == "annotation_xml"
    assert seg_payload["task_type"] == "classification_single_label_with_bbox"
    assert seg_payload["segmentation_supported"] is True
    assert seg_payload["bbox_count"] >= 1
    assert seg_payload["mask_data_url"].startswith("data:image/png;base64,")

    forbidden = client.post(
        "/api/v1/datasets/industrial/training/run",
        json={
            "dataset_name": "neu_surface",
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=viewer_headers,
    )
    assert forbidden.status_code == 403

    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")
    run_response = client.post(
        "/api/v1/datasets/industrial/training/run",
        json={
            "dataset_name": "neu_surface",
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=analyst_headers,
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["run_id"] is not None
    assert run_payload["dataset_name"] == "neu_surface"
    assert run_payload["storage"] == "sql"
    assert run_payload["task_type"] == "classification_single_label_with_bbox"
    assert run_payload["classification_mode"] in {"binary", "multiclass"}
    assert run_payload["label_source"] == "folder_name_plus_xml_bbox"
    assert run_payload["segmentation_supported"] is True
    assert run_payload["train_samples"] > 0
    assert run_payload["test_samples"] > 0
    assert run_payload["annotated_samples"] >= 1
    assert len(run_payload["curve"]) == 4
    assert 0.0 <= run_payload["train_accuracy"] <= 1.0
    assert 0.0 <= run_payload["test_accuracy"] <= 1.0

    latest_after = client.get(
        "/api/v1/datasets/industrial/training/latest",
        params={"dataset_name": "neu_surface"},
        headers=viewer_headers,
    )
    assert latest_after.status_code == 200
    latest_after_payload = latest_after.json()
    assert latest_after_payload["run_id"] == run_payload["run_id"]
    assert latest_after_payload["curve"]


def test_classification_only_datasets_report_segmentation_unavailable(client: TestClient) -> None:
    headers = _auth_headers(client, username="viewer", password="viewer123")

    steel_seg = client.get(
        "/api/v1/datasets/industrial/segmentation/steel_defect/train/crazing/0",
        headers=headers,
    )
    assert steel_seg.status_code == 200
    steel_payload = steel_seg.json()
    assert steel_payload["task_type"] == "classification_single_label"
    assert steel_payload["segmentation_supported"] is False
    assert steel_payload["source"] == "none"
    assert steel_payload["bbox_count"] == 0
    assert "Thumbs.db" in steel_payload["message"]

    casting_seg = client.get(
        "/api/v1/datasets/industrial/segmentation/casting_defect/train/def_front/0",
        headers=headers,
    )
    assert casting_seg.status_code == 200
    casting_payload = casting_seg.json()
    assert casting_payload["task_type"] == "classification_single_label"
    assert casting_payload["segmentation_supported"] is False
    assert casting_payload["source"] == "none"
    assert casting_payload["bbox_count"] == 0
    assert "classification-only" in casting_payload["message"]

    casting_latest = client.get(
        "/api/v1/datasets/industrial/training/latest",
        params={"dataset_name": "casting_defect"},
        headers=headers,
    )
    assert casting_latest.status_code == 200
    casting_latest_payload = casting_latest.json()
    assert casting_latest_payload["task_type"] == "classification_single_label"
    assert casting_latest_payload["classification_mode"] == "binary"
    assert casting_latest_payload["segmentation_supported"] is False
    assert casting_latest_payload["label_source"] == "folder_name"


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
    assert resync_payload["industrial_rows_synced"] >= 0
    assert len(recent_payload) >= 1
    assert "request_id" in recent_payload[0]


def test_metrics_endpoint_exposes_prometheus_counters(client: TestClient) -> None:
    # Trigger a few API calls before reading metrics.
    headers = _auth_headers(client)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    summary = client.get("/api/v1/dashboard/summary", headers=headers)
    assert summary.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inphase_http_requests_total" in response.text
    assert "inphase_http_request_duration_seconds" in response.text


def test_async_learning_job_queue_runs_training(client: TestClient) -> None:
    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")

    enqueue = client.post(
        "/api/v1/learning/jobs/busi-training",
        json={
            "include_normal": False,
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=analyst_headers,
    )
    assert enqueue.status_code == 200
    job_id = int(enqueue.json()["job_id"])

    status_payload = None
    for _ in range(60):
        status_response = client.get(f"/api/v1/learning/jobs/{job_id}", headers=analyst_headers)
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["result"] is not None
    assert status_payload["result"]["epochs"] == 3

    latest = client.get("/api/v1/datasets/busi/training/latest", headers=analyst_headers)
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["run_id"] is not None


def test_async_industrial_training_job_queue_runs_training(client: TestClient) -> None:
    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")

    enqueue = client.post(
        "/api/v1/learning/jobs/industrial-training",
        json={
            "dataset_name": "casting_defect",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.02,
        },
        headers=analyst_headers,
    )
    assert enqueue.status_code == 200
    enqueue_payload = enqueue.json()
    assert enqueue_payload["job_type"] == "industrial_training"
    job_id = int(enqueue_payload["job_id"])

    status_payload = None
    for _ in range(80):
        status_response = client.get(f"/api/v1/learning/jobs/{job_id}", headers=analyst_headers)
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["result"] is not None
    assert status_payload["result"]["dataset_name"] == "casting_defect"
    assert status_payload["result"]["epochs"] == 3

    latest = client.get(
        "/api/v1/datasets/industrial/training/latest",
        params={"dataset_name": "casting_defect"},
        headers=analyst_headers,
    )
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["run_id"] is not None
    assert latest_payload["dataset_name"] == "casting_defect"


def test_async_resync_job_queue_admin_only(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")
    admin_headers = _auth_headers(client, username="admin", password="admin123")

    forbidden = client.post("/api/v1/learning/jobs/datasets-resync", headers=viewer_headers)
    assert forbidden.status_code == 403

    enqueue = client.post("/api/v1/learning/jobs/datasets-resync", headers=admin_headers)
    assert enqueue.status_code == 200
    job_id = int(enqueue.json()["job_id"])

    status_payload = None
    for _ in range(80):
        status_response = client.get(f"/api/v1/learning/jobs/{job_id}", headers=admin_headers)
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["result"] is not None
    assert int(status_payload["result"]["busi_rows_synced"]) >= 0


def test_ops_database_schema_status_admin_only(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")
    admin_headers = _auth_headers(client, username="admin", password="admin123")

    forbidden = client.get("/api/v1/ops/database/schema-status", headers=viewer_headers)
    assert forbidden.status_code == 403

    # Touch industrial endpoint first so lazy SQL seeding is reflected in row counts.
    summary = client.get("/api/v1/datasets/industrial/summary", headers=admin_headers)
    assert summary.status_code == 200

    response = client.get("/api/v1/ops/database/schema-status", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()

    assert "database_url" in payload
    assert "alembic_current_revision" in payload
    assert "alembic_head_revision" in payload
    assert isinstance(payload["tables"], list)

    table_counts = {row["table_name"]: row["row_count"] for row in payload["tables"]}
    assert "industrial_samples" in table_counts
    assert "industrial_training_runs" in table_counts
    assert table_counts["industrial_samples"] >= 1
    assert table_counts["industrial_training_runs"] >= 0


def test_upload_endpoints_store_busi_and_industrial_samples_in_sql(client: TestClient) -> None:
    analyst_headers = _auth_headers(client, username="analyst", password="analyst123")

    image_rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    image_rgb[:, :, 1] = 180
    mask_gray = np.zeros((48, 48), dtype=np.uint8)
    mask_gray[12:30, 14:34] = 255

    img_buffer = BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(img_buffer, format="PNG")
    mask_buffer = BytesIO()
    Image.fromarray(mask_gray, mode="L").save(mask_buffer, format="PNG")

    busi_upload = client.post(
        "/api/v1/datasets/busi/upload",
        data={"class_name": "benign", "split": "train"},
        files={
            "image": ("uploaded_case.png", img_buffer.getvalue(), "image/png"),
            "mask": ("uploaded_case_mask.png", mask_buffer.getvalue(), "image/png"),
        },
        headers=analyst_headers,
    )
    assert busi_upload.status_code == 200
    busi_payload = busi_upload.json()
    assert busi_payload["storage"] == "sql"
    assert busi_payload["class_name"] == "benign"
    assert busi_payload["total_class_samples"] >= 1

    preview = client.get("/api/v1/datasets/busi/samples/benign/0", headers=analyst_headers)
    assert preview.status_code == 200
    assert preview.json()["total_samples"] >= 1

    industrial_buffer = BytesIO()
    Image.fromarray(np.full((42, 42, 3), 90, dtype=np.uint8), mode="RGB").save(
        industrial_buffer, format="PNG"
    )
    xml_blob = b"<annotation><object><name>crazing</name></object></annotation>"

    industrial_upload = client.post(
        "/api/v1/datasets/industrial/upload",
        data={
            "dataset_name": "neu_surface",
            "split": "train",
            "class_name": "crazing",
        },
        files={
            "image": ("neu_uploaded.png", industrial_buffer.getvalue(), "image/png"),
            "annotation": ("neu_uploaded.xml", xml_blob, "application/xml"),
        },
        headers=analyst_headers,
    )
    assert industrial_upload.status_code == 200
    industrial_payload = industrial_upload.json()
    assert industrial_payload["storage"] == "sql"
    assert industrial_payload["dataset_name"] == "neu_surface"
    assert industrial_payload["has_annotation"] is True

    summary = client.get("/api/v1/datasets/industrial/summary", headers=analyst_headers)
    assert summary.status_code == 200
    summary_payload = summary.json()
    assert summary_payload["totals_by_dataset"]["neu_surface"] >= 1


def test_upload_endpoints_require_analyst_role(client: TestClient) -> None:
    viewer_headers = _auth_headers(client, username="viewer", password="viewer123")

    img_buffer = BytesIO()
    Image.fromarray(np.full((24, 24, 3), 30, dtype=np.uint8), mode="RGB").save(
        img_buffer, format="PNG"
    )

    forbidden_busi = client.post(
        "/api/v1/datasets/busi/upload",
        data={"class_name": "benign", "split": "train"},
        files={"image": ("forbidden.png", img_buffer.getvalue(), "image/png")},
        headers=viewer_headers,
    )
    assert forbidden_busi.status_code == 403

    forbidden_industrial = client.post(
        "/api/v1/datasets/industrial/upload",
        data={
            "dataset_name": "steel_defect",
            "split": "train",
            "class_name": "crazing",
        },
        files={"image": ("forbidden.png", img_buffer.getvalue(), "image/png")},
        headers=viewer_headers,
    )
    assert forbidden_industrial.status_code == 403
