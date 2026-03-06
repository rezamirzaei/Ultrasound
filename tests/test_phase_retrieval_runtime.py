"""Runtime tests for PICMUS phase retrieval services and dataset download paths."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from tests._picmus_fixtures import create_picmus_fixture
from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import PhaseRetrievalPreviewRequest
from ultrasound.api.services.phase_retrieval_service import PhaseRetrievalService
from ultrasound.api.services.service_errors import DependencyUnavailableError, InvalidRequestError
from ultrasound.data.picmus_dataset import (
    PICMUS_IN_VIVO_URL,
    download_picmus_in_vivo,
    picmus_in_vivo_available,
    select_high_energy_rf_segment,
)
from ultrasound.workflows import run_phase_retrieval_picmus, run_phase_retrieval_ultrasound


class _FakeResponse(BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _build_picmus_zip_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(
            "in_vivo/carotid_long/carotid_long_expe_dataset_rf.hdf5",
            b"placeholder-hdf5-content",
        )
    return buffer.getvalue()


def _service_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        project_root=tmp_path,
        data_dir=data_dir,
        busi_dir=data_dir / "busi",
        ndt_dir=data_dir / "ascan_signals" / "ndt_samples",
        ui_dir=tmp_path / "ui",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_download_picmus_in_vivo_extracts_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _build_picmus_zip_bytes()
    requested_urls: list[str] = []

    def _fake_urlopen(url: str, timeout: float = 0.0) -> _FakeResponse:
        requested_urls.append(url)
        assert timeout == 120.0
        return _FakeResponse(payload)

    monkeypatch.setattr("ultrasound.data.picmus_dataset.urlopen", _fake_urlopen)

    root = download_picmus_in_vivo(tmp_path / "picmus")

    assert requested_urls == [PICMUS_IN_VIVO_URL]
    assert (tmp_path / "picmus" / "in_vivo.zip").exists()
    assert root == (tmp_path / "picmus" / "in_vivo")
    assert (root / "carotid_long" / "carotid_long_expe_dataset_rf.hdf5").exists()


def test_phase_retrieval_service_reports_missing_dataset(tmp_path: Path) -> None:
    service = PhaseRetrievalService(_service_config(tmp_path))

    status = service.get_status()

    assert status.dataset_available is False
    assert status.recommended_case == "carotid_long"
    with pytest.raises(DependencyUnavailableError):
        service.preview(PhaseRetrievalPreviewRequest())


def test_phase_retrieval_service_validates_unknown_case(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path / "data")
    service = PhaseRetrievalService(_service_config(tmp_path))

    with pytest.raises(InvalidRequestError):
        service.preview(
            PhaseRetrievalPreviewRequest(
                case_name="not-a-case",
                segment_length=96,
                measurement_ratio=5,
                max_iterations=150,
                seed=0,
            )
        )


def test_phase_retrieval_service_preview_accepts_explicit_segment_indices(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path / "data")
    service = PhaseRetrievalService(_service_config(tmp_path))

    response = service.preview(
        PhaseRetrievalPreviewRequest(
            case_name="carotid_long",
            segment_length=96,
            measurement_ratio=5,
            max_iterations=150,
            seed=7,
            angle_index=1,
            element_index=2,
            start_index=72,
        )
    )

    assert response.case_name == "carotid_long"
    assert response.angle_index == 1
    assert response.element_index == 2
    assert response.start_index == 72
    assert response.overall_pass is True


def test_select_high_energy_rf_segment_validates_length(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path)

    with pytest.raises(ValueError):
        select_high_energy_rf_segment(
            tmp_path / "picmus",
            case_name="carotid_cross",
            segment_length=1000,
        )


def test_run_phase_retrieval_ultrasound_rejects_invalid_inputs() -> None:
    rf = np.zeros(64, dtype=np.float64)

    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(rf, measurement_ratio=0)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(rf, n_iter=0)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(rf, step_size=0.0)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(rf, noise_scale=-0.1)


def test_run_phase_retrieval_picmus_requires_indices_for_explicit_start(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path)

    with pytest.raises(ValueError):
        run_phase_retrieval_picmus(
            root_dir=str(tmp_path / "picmus"),
            case_name="carotid_cross",
            start_index=108,
        )


def test_picmus_fixture_marks_dataset_available(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path)

    assert picmus_in_vivo_available(tmp_path / "picmus") is True
