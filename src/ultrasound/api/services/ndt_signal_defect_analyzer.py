"""Signal-derived defect candidate analysis for NDT A-scan data."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ultrasound.api.models.domain import NdtAnalyzedDefect, NdtSampleRecord, NdtSignalAnalysisRecord
from ultrasound.api.services.ndt_peak_detector import NdtPeakDetector


class NdtSignalDefectAnalyzer:
    """Propose defect candidates directly from waveform peaks."""

    def __init__(self, peak_detector: NdtPeakDetector) -> None:
        self.peak_detector = peak_detector

    def detect(
        self,
        sample: NdtSampleRecord,
        signal_analysis: NdtSignalAnalysisRecord,
    ) -> list[NdtAnalyzedDefect]:
        rf = np.asarray(sample.rf, dtype=np.float64).reshape(-1)
        time_s = np.asarray(sample.time, dtype=np.float64).reshape(-1)
        if rf.size < 16:
            return []

        envelope = np.abs(hilbert(rf))
        (
            envelope_smooth,
            peak_indices,
            peak_prominences,
            peak_height,
            _,
            min_peak_distance,
        ) = self.peak_detector.detect_peaks(sample, envelope)
        if peak_indices.size == 0:
            return []

        front_wall_idx = (
            int(signal_analysis.front_wall.index)
            if signal_analysis.front_wall is not None
            else None
        )
        back_wall_idx = (
            int(signal_analysis.back_wall.index) if signal_analysis.back_wall is not None else None
        )
        if front_wall_idx is None:
            return []

        max_envelope = max(float(np.max(envelope_smooth)), 1e-12)
        prominence_scale = max(peak_height, 1e-12)
        gate_distance = max(min_peak_distance, int(round(0.25e-6 * float(sample.fs_hz))))

        candidates: list[NdtAnalyzedDefect] = []
        for idx, peak_idx in enumerate(peak_indices):
            peak_idx_int = int(peak_idx)
            if peak_idx_int <= front_wall_idx:
                continue
            if abs(peak_idx_int - front_wall_idx) <= gate_distance:
                continue
            if back_wall_idx is not None and peak_idx_int >= back_wall_idx:
                continue
            if back_wall_idx is not None and abs(peak_idx_int - back_wall_idx) <= gate_distance:
                continue

            depth_m = self._estimate_depth(
                time_s=time_s,
                peak_idx=peak_idx_int,
                front_wall_idx=front_wall_idx,
                back_wall_idx=back_wall_idx,
                c_mps=float(sample.c_mps),
                thickness_m=sample.thickness_m,
            )
            if depth_m is None or depth_m <= 0.0:
                continue
            if sample.thickness_m is not None and depth_m >= 0.98 * float(sample.thickness_m):
                continue

            amplitude = float(envelope_smooth[peak_idx_int])
            amp_norm = float(np.clip(amplitude / max_envelope, 0.0, 1.0))
            prominence_norm = float(np.clip(peak_prominences[idx] / prominence_scale, 0.0, 1.0))
            confidence = float(np.clip(0.45 * amp_norm + 0.55 * prominence_norm, 0.0, 1.0))
            if confidence < 0.25:
                continue

            candidates.append(
                NdtAnalyzedDefect(
                    depth_m=float(depth_m),
                    amplitude=amplitude,
                    time_us=self.peak_detector.refine_peak_time_us(
                        envelope_smooth,
                        time_s,
                        peak_idx_int,
                    ),
                    confidence=confidence,
                    source="signal",
                )
            )

        return candidates

    @staticmethod
    def _estimate_depth(
        time_s: np.ndarray,
        peak_idx: int,
        front_wall_idx: int,
        back_wall_idx: int | None,
        c_mps: float,
        thickness_m: float | None,
    ) -> float | None:
        peak_time = float(time_s[peak_idx])
        front_time = float(time_s[front_wall_idx])
        if peak_time <= front_time:
            return None

        if (
            thickness_m is not None
            and back_wall_idx is not None
            and float(time_s[back_wall_idx]) > front_time
        ):
            back_time = float(time_s[back_wall_idx])
            ratio = (peak_time - front_time) / (back_time - front_time)
            return float(np.clip(ratio, 0.0, 1.0) * float(thickness_m))

        if c_mps <= 0.0:
            return None
        return 0.5 * c_mps * (peak_time - front_time)
