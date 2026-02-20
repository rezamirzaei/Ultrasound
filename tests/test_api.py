"""API integration tests for REST endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ultrasound.api.app import create_app


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_endpoint() -> None:
    client = _client()
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]


def test_root_redirects_to_ui() -> None:
    client = _client()
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/index.html"


def test_ui_index_served() -> None:
    client = _client()
    response = client.get("/ui/index.html")

    assert response.status_code == 200
    assert "inPhase Ultrasound Platform" in response.text


def test_dashboard_summary_endpoint() -> None:
    client = _client()
    response = client.get("/api/v1/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "busi_counts" in payload
    assert "busi_total" in payload
    assert payload["busi_total"] >= 0


def test_ndt_sample_listing_and_detail() -> None:
    client = _client()
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


def test_preprocessing_preview_endpoint() -> None:
    client = _client()
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
