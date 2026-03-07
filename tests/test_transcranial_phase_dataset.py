"""Tests for the transcranial hydrophone dataset helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import py7zr
import pytest

from tests._transcranial_fixtures import create_transcranial_fixture
from ultrasound.data.transcranial_phase_dataset import (
    ETH_TRANSCRANIAL_DATA_URL,
    default_transcranial_case,
    download_transcranial_dataset,
    list_transcranial_scan_cases,
    load_transcranial_waveform_window,
    resolve_transcranial_dataset_root,
    select_high_energy_hydrophone_window,
    transcranial_dataset_available,
)


class _FakeResponse(BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _build_transcranial_archive_bytes(tmp_path: Path) -> bytes:
    source_root = create_transcranial_fixture(tmp_path / "source")
    archive_path = tmp_path / "transcranial-fixture.7z"
    with py7zr.SevenZipFile(archive_path, mode="w") as archive:
        for mat_path in sorted((source_root / "Scan_data").glob("*.mat")):
            archive.write(mat_path, arcname=f"Scan_data/{mat_path.name}")
    return archive_path.read_bytes()


def test_transcranial_dataset_helpers_resolve_cases_and_windows(tmp_path: Path) -> None:
    root = create_transcranial_fixture(tmp_path)

    assert resolve_transcranial_dataset_root(tmp_path / "phase_retrieval") == root
    assert transcranial_dataset_available(tmp_path / "phase_retrieval") is True
    assert default_transcranial_case(tmp_path / "phase_retrieval") == "Parietal_free_field_0_XY"
    assert "Frontal_40_XY" in list_transcranial_scan_cases(tmp_path / "phase_retrieval")

    selected = select_high_energy_hydrophone_window(
        tmp_path / "phase_retrieval",
        case_name="Parietal_free_field_0_XY",
        window_length=256,
    )
    assert selected.row_index == 15
    assert selected.col_index == 20
    assert selected.window_length == 256
    assert selected.trace_energy > 0.0
    assert selected.dominant_frequency_bin >= 1
    assert selected.scan_energy_map.shape == (32, 32)


def test_load_transcranial_waveform_window_validates_bounds(tmp_path: Path) -> None:
    create_transcranial_fixture(tmp_path)

    explicit = load_transcranial_waveform_window(
        tmp_path / "phase_retrieval",
        case_name="Frontal_40_XY",
        row_index=12,
        col_index=24,
        start_index=120,
        window_length=192,
    )
    assert explicit.row_index == 12
    assert explicit.col_index == 24
    assert explicit.trace.shape == (192,)

    with pytest.raises(ValueError):
        load_transcranial_waveform_window(
            tmp_path / "phase_retrieval",
            case_name="Frontal_40_XY",
            row_index=99,
            col_index=0,
            start_index=0,
            window_length=192,
        )

    with pytest.raises(ValueError):
        load_transcranial_waveform_window(
            tmp_path / "phase_retrieval",
            case_name="Frontal_40_XY",
            row_index=0,
            col_index=0,
            start_index=999,
            window_length=192,
        )


def test_download_transcranial_dataset_extracts_scan_archive(
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
    assert root == (tmp_path / "phase_retrieval" / "transcranial")
    assert (root / "Scan_data" / "Parietal_free_field_0_XY.mat").exists()
    assert transcranial_dataset_available(tmp_path / "phase_retrieval") is True
