"""Runtime tests for transcranial phase retrieval services and dataset download paths."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import py7zr
import pytest

from tests._transcranial_fixtures import create_transcranial_fixture
from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import PhaseRetrievalPreviewRequest
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.phase_retrieval_service import PhaseRetrievalService
from ultrasound.api.services.service_errors import DependencyUnavailableError, InvalidRequestError
from ultrasound.data.transcranial_phase_dataset import (
    ETH_TRANSCRANIAL_DATA_URL,
    download_transcranial_dataset,
    select_high_energy_hydrophone_window,
    transcranial_dataset_available,
)
from ultrasound.workflows import run_phase_retrieval_transcranial, run_phase_retrieval_ultrasound


class _FakeResponse(BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _build_transcranial_archive_bytes(tmp_path: Path) -> bytes:
    source_root = create_transcranial_fixture(tmp_path / "source")
    archive_path = tmp_path / "transcranial-runtime.7z"
    with py7zr.SevenZipFile(archive_path, mode="w") as archive:
        for mat_path in sorted((source_root / "Scan_data").glob("*.mat")):
            archive.write(mat_path, arcname=f"Scan_data/{mat_path.name}")
    return archive_path.read_bytes()


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


def test_download_transcranial_dataset_extracts_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _build_transcranial_archive_bytes(tmp_path)
    requested_urls: list[str] = []

    def _fake_urlopen(url: str, timeout: float = 0.0) -> _FakeResponse:
        requested_urls.append(url)
        assert timeout == 180.0
        return _FakeResponse(payload)

    monkeypatch.setattr("ultrasound.data.transcranial_phase_dataset.urlopen", _fake_urlopen)

    root = download_transcranial_dataset(tmp_path / "phase_retrieval")

    assert requested_urls == [ETH_TRANSCRANIAL_DATA_URL]
    assert (tmp_path / "phase_retrieval" / "Transcranial_data.7z").exists()
    assert root == (tmp_path / "phase_retrieval" / "transcranial")
    assert (root / "Scan_data" / "Frontal_40_XY.mat").exists()


def test_phase_retrieval_service_reports_missing_dataset(tmp_path: Path) -> None:
    service = PhaseRetrievalService(_service_config(tmp_path), MediaService())

    status = service.get_status()

    assert status.dataset_available is False
    assert status.recommended_case == "Parietal_free_field_0_XY"
    assert "hydrophone pulse" in status.recovered_quantity.lower()
    with pytest.raises(DependencyUnavailableError):
        service.preview(PhaseRetrievalPreviewRequest())


def test_phase_retrieval_service_validates_unknown_case(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path / "data")
    service = PhaseRetrievalService(_service_config(tmp_path), MediaService())

    with pytest.raises(InvalidRequestError):
        service.preview(
            PhaseRetrievalPreviewRequest(
                case_name="not-a-case",
                window_length=256,
                n_fft=80,
                hop_length=8,
                max_iterations=120,
                seed=0,
            )
        )


def test_phase_retrieval_service_preview_accepts_explicit_window_indices(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path / "data")
    service = PhaseRetrievalService(_service_config(tmp_path), MediaService())
    selected = select_high_energy_hydrophone_window(
        tmp_path / "data" / "phase_retrieval",
        case_name="Parietal_free_field_0_XY",
        window_length=256,
    )

    response = service.preview(
        PhaseRetrievalPreviewRequest(
            case_name="Parietal_free_field_0_XY",
            window_length=256,
            n_fft=80,
            hop_length=8,
            max_iterations=120,
            seed=7,
            row_index=selected.row_index,
            col_index=selected.col_index,
            start_index=selected.start_index,
        )
    )

    assert response.case_name == "Parietal_free_field_0_XY"
    assert response.row_index == selected.row_index
    assert response.col_index == selected.col_index
    assert response.start_index == selected.start_index
    assert response.overall_pass is True
    assert response.scan_image_data_url.startswith("data:image/png;base64,")
    assert response.spectrogram_image_data_url.startswith("data:image/png;base64,")


def test_select_high_energy_hydrophone_window_validates_length(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    with pytest.raises(ValueError):
        select_high_energy_hydrophone_window(
            tmp_path / "phase_retrieval",
            case_name="Frontal_free_field_0_XY",
            window_length=999,
        )


def test_run_phase_retrieval_ultrasound_rejects_invalid_inputs() -> None:
    waveform = np.zeros(128, dtype=np.float64)

    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(waveform, n_fft=8)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(waveform, hop_length=80, n_fft=80)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(waveform, n_iter=0)
    with pytest.raises(ValueError):
        run_phase_retrieval_ultrasound(np.zeros(32, dtype=np.float64), n_fft=64)


def test_run_phase_retrieval_transcranial_requires_indices_for_explicit_start(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    with pytest.raises(ValueError):
        run_phase_retrieval_transcranial(
            root_dir=str(tmp_path / "phase_retrieval"),
            case_name="Parietal_free_field_0_XY",
            start_index=108,
        )


def test_transcranial_fixture_marks_dataset_available(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    assert transcranial_dataset_available(tmp_path / "phase_retrieval") is True
