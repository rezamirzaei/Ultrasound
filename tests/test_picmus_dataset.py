"""Tests for PICMUS real-ultrasound dataset helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._picmus_fixtures import create_picmus_fixture
from ultrasound.data.picmus_dataset import (
    default_picmus_case,
    list_picmus_rf_cases,
    load_picmus_rf_segment,
    picmus_in_vivo_available,
    resolve_picmus_in_vivo_root,
    select_high_energy_rf_segment,
)


def test_picmus_dataset_helpers_resolve_cases_and_segments(tmp_path: Path) -> None:
    root = create_picmus_fixture(tmp_path)

    assert resolve_picmus_in_vivo_root(tmp_path / "picmus") == root
    assert picmus_in_vivo_available(tmp_path / "picmus") is True
    assert list_picmus_rf_cases(tmp_path / "picmus") == ["carotid_cross", "carotid_long"]
    assert default_picmus_case(tmp_path / "picmus") == "carotid_cross"

    segment = select_high_energy_rf_segment(
        tmp_path / "picmus",
        case_name="carotid_cross",
        segment_length=96,
    )

    assert segment.case_name == "carotid_cross"
    assert segment.angle_index == 0
    assert segment.element_index == 1
    assert segment.start_index == 108
    assert segment.segment_length == 96
    assert segment.energy > 10.0


def test_load_picmus_rf_segment_validates_bounds(tmp_path: Path) -> None:
    create_picmus_fixture(tmp_path)

    explicit = load_picmus_rf_segment(
        tmp_path / "picmus",
        case_name="carotid_long",
        angle_index=1,
        element_index=2,
        start_index=72,
        segment_length=96,
    )
    assert explicit.segment_length == 96
    assert explicit.rf_segment.shape == (96,)

    with pytest.raises(ValueError):
        load_picmus_rf_segment(
            tmp_path / "picmus",
            case_name="carotid_long",
            angle_index=10,
            element_index=0,
            start_index=0,
            segment_length=32,
        )
