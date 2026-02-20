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
        defects=np.array([[0.005, 0.4]], dtype=np.float64),
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
