"""Thickness and wall-echo analysis for NDT A-scan signals."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from scipy.signal import hilbert

from ultrasound.api.models.domain import (
    NdtSampleRecord,
    NdtSignalAnalysisRecord,
    NdtWallEchoRecord,
)
from ultrasound.api.services.ndt_peak_detector import NdtPeakDetector

ThicknessMethod = Literal["time_of_flight", "absolute_backwall", "insufficient_data"]


class NdtThicknessAnalyzer:
    """Analyze waveform peaks and derive wall echoes plus thickness metrics."""

    def __init__(self, peak_detector: NdtPeakDetector) -> None:
        self.peak_detector = peak_detector

    def analyze_signal(self, sample: NdtSampleRecord) -> NdtSignalAnalysisRecord:
        rf = np.asarray(sample.rf, dtype=np.float64).reshape(-1)
        time_s = np.asarray(sample.time, dtype=np.float64).reshape(-1)
        if rf.size < 16:
            return NdtSignalAnalysisRecord()

        envelope = np.abs(hilbert(rf))
        (
            envelope_smooth,
            peak_indices,
            peak_prominences,
            peak_height,
            noise_sigma,
            _,
        ) = self.peak_detector.detect_peaks(sample, envelope)
        if peak_indices.size == 0:
            return NdtSignalAnalysisRecord()

        prominence_by_idx = {
            int(peak_indices[i]): float(peak_prominences[i]) for i in range(peak_indices.size)
        }
        dt_us = self.peak_detector.sample_period_us(time_s)

        front_wall_idx = self.peak_detector.find_front_wall(peak_indices, time_s)
        back_wall_idx = None
        if front_wall_idx is not None:
            back_wall_idx = self.peak_detector.find_back_wall(
                sample=sample,
                peak_indices=peak_indices,
                time_s=time_s,
                envelope_smooth=envelope_smooth,
                front_wall_idx=front_wall_idx,
            )

        front_wall = None
        front_time_us = None
        front_std_us = None
        front_confidence = None
        if front_wall_idx is not None:
            front_time_us = self.peak_detector.refine_peak_time_us(
                envelope_smooth,
                time_s,
                int(front_wall_idx),
            )
            front_std_us = self.peak_detector.estimate_time_std_us(
                envelope_smooth=envelope_smooth,
                peak_idx=int(front_wall_idx),
                prominence=prominence_by_idx.get(int(front_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
                dt_us=dt_us,
            )
            front_confidence = self.peak_detector.echo_confidence(
                amplitude=float(envelope_smooth[int(front_wall_idx)]),
                prominence=prominence_by_idx.get(int(front_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
            )
            front_wall = NdtWallEchoRecord(
                label="front_wall",
                index=int(front_wall_idx),
                time_us=front_time_us,
                depth_m=0.0,
                amplitude=float(envelope_smooth[int(front_wall_idx)]),
                confidence=front_confidence,
                time_std_us=front_std_us,
            )

        back_wall = None
        back_time_us = None
        back_std_us = None
        back_confidence = None
        if back_wall_idx is not None:
            back_time_us = self.peak_detector.refine_peak_time_us(
                envelope_smooth,
                time_s,
                int(back_wall_idx),
            )
            back_std_us = self.peak_detector.estimate_time_std_us(
                envelope_smooth=envelope_smooth,
                peak_idx=int(back_wall_idx),
                prominence=prominence_by_idx.get(int(back_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
                dt_us=dt_us,
            )
            back_confidence = self.peak_detector.echo_confidence(
                amplitude=float(envelope_smooth[int(back_wall_idx)]),
                prominence=prominence_by_idx.get(int(back_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
            )

        nominal_thickness_mm = (
            float(sample.thickness_m * 1e3) if sample.thickness_m is not None else None
        )
        (
            estimated_thickness_mm,
            thickness_std_mm,
            thickness_ci95_lower_mm,
            thickness_ci95_upper_mm,
            thickness_confidence,
            thickness_method,
        ) = self._resolve_thickness(
            c_mps=float(sample.c_mps),
            front_time_us=front_time_us,
            back_time_us=back_time_us,
            front_std_us=front_std_us,
            back_std_us=back_std_us,
            front_confidence=front_confidence,
            back_confidence=back_confidence,
            nominal_thickness_mm=nominal_thickness_mm,
        )

        if back_wall_idx is not None:
            back_wall = NdtWallEchoRecord(
                label="back_wall",
                index=int(back_wall_idx),
                time_us=(
                    float(back_time_us)
                    if back_time_us is not None
                    else float(time_s[int(back_wall_idx)] * 1e6)
                ),
                depth_m=(
                    float(estimated_thickness_mm / 1e3)
                    if estimated_thickness_mm is not None
                    else None
                ),
                amplitude=float(envelope_smooth[int(back_wall_idx)]),
                confidence=back_confidence,
                time_std_us=back_std_us,
            )

        thickness_error_mm = None
        if estimated_thickness_mm is not None and nominal_thickness_mm is not None:
            thickness_error_mm = float(estimated_thickness_mm - nominal_thickness_mm)

        return NdtSignalAnalysisRecord(
            total_peaks=int(peak_indices.size),
            peak_indices=[int(v) for v in peak_indices],
            peak_times_us=[float(time_s[int(v)] * 1e6) for v in peak_indices],
            front_wall=front_wall,
            back_wall=back_wall,
            estimated_thickness_mm=estimated_thickness_mm,
            thickness_std_mm=thickness_std_mm,
            thickness_ci95_lower_mm=thickness_ci95_lower_mm,
            thickness_ci95_upper_mm=thickness_ci95_upper_mm,
            thickness_confidence=thickness_confidence,
            thickness_method=thickness_method,
            nominal_thickness_mm=nominal_thickness_mm,
            thickness_error_mm=thickness_error_mm,
            thinning_flag=self._compute_thinning_flag(
                sample=sample,
                estimated_thickness_mm=estimated_thickness_mm,
            ),
        )

    def _resolve_thickness(
        self,
        c_mps: float,
        front_time_us: float | None,
        back_time_us: float | None,
        front_std_us: float | None,
        back_std_us: float | None,
        front_confidence: float | None,
        back_confidence: float | None,
        nominal_thickness_mm: float | None,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        ThicknessMethod,
    ]:
        if c_mps <= 0.0 or back_time_us is None:
            return None, None, None, None, None, "insufficient_data"

        candidates: list[tuple[ThicknessMethod, float, float | None]] = []

        abs_thickness_mm = 0.5 * c_mps * back_time_us * 1e-3
        abs_std_mm = 0.5 * c_mps * float(back_std_us or 0.0) * 1e-3 if back_std_us else None
        candidates.append(("absolute_backwall", float(abs_thickness_mm), abs_std_mm))

        if front_time_us is not None and back_time_us > front_time_us:
            tof_us = back_time_us - front_time_us
            tof_thickness_mm = 0.5 * c_mps * tof_us * 1e-3
            tof_std_us = None
            if front_std_us is not None and back_std_us is not None:
                tof_std_us = float(np.sqrt(front_std_us**2 + back_std_us**2))
            tof_std_mm = 0.5 * c_mps * tof_std_us * 1e-3 if tof_std_us is not None else None
            candidates.append(("time_of_flight", float(tof_thickness_mm), tof_std_mm))

        if nominal_thickness_mm is not None:
            method, estimate_mm, std_mm = min(
                candidates,
                key=lambda item: abs(item[1] - nominal_thickness_mm),
            )
        else:
            tof_candidate = [item for item in candidates if item[0] == "time_of_flight"]
            method, estimate_mm, std_mm = tof_candidate[0] if tof_candidate else candidates[0]

        ci_lower_mm = None
        ci_upper_mm = None
        if std_mm is not None:
            ci_lower_mm = max(0.0, estimate_mm - 1.96 * std_mm)
            ci_upper_mm = max(ci_lower_mm, estimate_mm + 1.96 * std_mm)

        wall_conf = (
            float(
                np.mean(
                    [value for value in (front_confidence, back_confidence) if value is not None]
                )
            )
            if any(value is not None for value in (front_confidence, back_confidence))
            else 0.5
        )
        if std_mm is not None and estimate_mm > 0.0:
            rel_unc = float(np.clip(std_mm / max(estimate_mm, 1e-9), 0.0, 1.0))
            thickness_conf = float(np.clip(wall_conf * np.exp(-2.2 * rel_unc), 0.0, 1.0))
        else:
            thickness_conf = float(np.clip(wall_conf * 0.85, 0.0, 1.0))

        return estimate_mm, std_mm, ci_lower_mm, ci_upper_mm, thickness_conf, method

    @staticmethod
    def _compute_thinning_flag(
        sample: NdtSampleRecord,
        estimated_thickness_mm: float | None,
    ) -> bool:
        if estimated_thickness_mm is None:
            return False

        context_text = f"{sample.name} {sample.description}"
        thinning_context = bool(
            re.search(r"corrosion|thinning|thinned|corrod", context_text, re.I)
        )
        reference_match = re.search(r"original\s*([0-9]+(?:\.[0-9]+)?)\s*mm", context_text, re.I)

        if reference_match:
            original_mm = float(reference_match.group(1))
            return estimated_thickness_mm < 0.95 * original_mm
        if thinning_context:
            return True
        return False
