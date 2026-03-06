"""Tests for signal-derived NDT defect candidate analysis."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from ultrasound.api.models.domain import NdtSampleRecord, NdtSignalAnalysisRecord, NdtWallEchoRecord
from ultrasound.api.services.ndt_peak_detector import NdtPeakDetector
from ultrasound.api.services.ndt_signal_defect_analyzer import NdtSignalDefectAnalyzer


class _PeakDetectorStub:
    def __init__(
        self,
        *,
        envelope_smooth: np.ndarray,
        peak_indices: np.ndarray,
        peak_prominences: np.ndarray,
        peak_height: float,
        min_peak_distance: int,
    ) -> None:
        self._envelope_smooth = envelope_smooth
        self._peak_indices = peak_indices
        self._peak_prominences = peak_prominences
        self._peak_height = peak_height
        self._min_peak_distance = min_peak_distance

    def detect_peaks(self, sample, envelope):  # noqa: ANN001
        return (
            self._envelope_smooth,
            self._peak_indices,
            self._peak_prominences,
            self._peak_height,
            None,
            self._min_peak_distance,
        )

    def refine_peak_time_us(self, envelope_smooth, time_s, peak_idx):  # noqa: ANN001
        return float(time_s[peak_idx] * 1e6)


def _sample(*, n_points: int = 64, c_mps: float = 5900.0, thickness_m: float | None = 0.02) -> NdtSampleRecord:
    time_s = np.linspace(0.0, 8e-6, n_points)
    rf = np.sin(np.linspace(0.0, 4.0 * np.pi, n_points))
    return NdtSampleRecord(
        name="sample",
        path=Path("sample.npz"),
        rf=rf,
        time=time_s,
        fs_hz=40e6,
        fc_hz=5e6,
        c_mps=c_mps,
        thickness_m=thickness_m,
        description="synthetic",
        defects=[],
    )


def _signal_analysis(front_idx: int | None, back_idx: int | None) -> NdtSignalAnalysisRecord:
    front_wall = (
        NdtWallEchoRecord(label="front_wall", index=front_idx, time_us=1.0)
        if front_idx is not None
        else None
    )
    back_wall = (
        NdtWallEchoRecord(label="back_wall", index=back_idx, time_us=6.0)
        if back_idx is not None
        else None
    )
    return NdtSignalAnalysisRecord(front_wall=front_wall, back_wall=back_wall)


def test_detect_returns_empty_for_short_rf() -> None:
    analyzer = NdtSignalDefectAnalyzer(
        cast(
            NdtPeakDetector,
            _PeakDetectorStub(
            envelope_smooth=np.ones(8, dtype=np.float64),
            peak_indices=np.array([4]),
            peak_prominences=np.array([1.0]),
            peak_height=1.0,
            min_peak_distance=1,
            ),
        )
    )

    result = analyzer.detect(_sample(n_points=8), _signal_analysis(front_idx=1, back_idx=6))

    assert result == []


def test_detect_returns_empty_without_peaks_or_front_wall() -> None:
    analyzer = NdtSignalDefectAnalyzer(
        cast(
            NdtPeakDetector,
            _PeakDetectorStub(
            envelope_smooth=np.ones(32, dtype=np.float64),
            peak_indices=np.array([], dtype=np.int64),
            peak_prominences=np.array([], dtype=np.float64),
            peak_height=1.0,
            min_peak_distance=2,
            ),
        )
    )

    no_peaks = analyzer.detect(_sample(n_points=32), _signal_analysis(front_idx=2, back_idx=20))
    no_front_wall = analyzer.detect(_sample(n_points=32), _signal_analysis(front_idx=None, back_idx=20))

    assert no_peaks == []
    assert no_front_wall == []


def test_detect_filters_invalid_peaks_and_returns_signal_candidate() -> None:
    sample = _sample()
    analyzer = NdtSignalDefectAnalyzer(
        cast(
            NdtPeakDetector,
            _PeakDetectorStub(
            envelope_smooth=np.linspace(0.1, 1.0, sample.rf.size),
            peak_indices=np.array([6, 20, 47, 55]),
            peak_prominences=np.array([0.3, 1.0, 0.8, 0.9]),
            peak_height=1.0,
            min_peak_distance=4,
            ),
        )
    )

    result = analyzer.detect(sample, _signal_analysis(front_idx=5, back_idx=48))

    assert len(result) == 1
    candidate = result[0]
    assert candidate.source == "signal"
    assert candidate.depth_m is not None
    assert sample.thickness_m is not None
    assert 0.0 < candidate.depth_m < sample.thickness_m
    assert candidate.confidence >= 0.25
    assert candidate.time_us is not None


def test_estimate_depth_uses_backwall_ratio_and_validates_velocity() -> None:
    time_s = np.array([0.0, 1e-6, 2e-6, 4e-6], dtype=np.float64)

    ratio_depth = NdtSignalDefectAnalyzer._estimate_depth(
        time_s=time_s,
        peak_idx=2,
        front_wall_idx=1,
        back_wall_idx=3,
        c_mps=5900.0,
        thickness_m=0.02,
    )
    invalid_velocity_depth = NdtSignalDefectAnalyzer._estimate_depth(
        time_s=time_s,
        peak_idx=2,
        front_wall_idx=1,
        back_wall_idx=None,
        c_mps=0.0,
        thickness_m=None,
    )

    assert ratio_depth == pytest.approx(0.02 / 3.0)
    assert invalid_velocity_depth is None
