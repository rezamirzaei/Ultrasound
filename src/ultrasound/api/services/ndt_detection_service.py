"""NDT waveform-based defect detection and metadata fusion."""

from __future__ import annotations

import re
from typing import Literal, Sequence

import numpy as np
from scipy.signal import find_peaks, hilbert, peak_widths

from ultrasound.api.models.domain import (
    NdtAnalyzedDefect,
    NdtSampleRecord,
    NdtSignalAnalysisRecord,
    NdtWallEchoRecord,
)

ThicknessMethod = Literal["time_of_flight", "absolute_backwall", "insufficient_data"]


class NdtDetectionService:
    """Detect and fuse NDT defects from signal content and metadata."""

    def __init__(
        self,
        min_relative_height: float = 0.22,
        noise_sigma_factor: float = 5.0,
        peak_distance_us: float = 0.35,
    ):
        self.min_relative_height = float(min_relative_height)
        self.noise_sigma_factor = float(noise_sigma_factor)
        self.peak_distance_us = float(peak_distance_us)

    def analyze_signal(self, sample: NdtSampleRecord) -> NdtSignalAnalysisRecord:
        """Analyze RF signal and return peak/wall/thickness descriptors."""
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
        ) = self._detect_peaks(sample, envelope)
        if peak_indices.size == 0:
            return NdtSignalAnalysisRecord()

        prominence_by_idx = {
            int(peak_indices[i]): float(peak_prominences[i]) for i in range(peak_indices.size)
        }
        dt_us = self._sample_period_us(time_s)

        front_wall_idx = self._find_front_wall(peak_indices, time_s)
        back_wall_idx = None
        if front_wall_idx is not None:
            back_wall_idx = self._find_back_wall(
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
            front_time_us = self._refine_peak_time_us(envelope_smooth, time_s, int(front_wall_idx))
            front_std_us = self._estimate_time_std_us(
                envelope_smooth=envelope_smooth,
                peak_idx=int(front_wall_idx),
                prominence=prominence_by_idx.get(int(front_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
                dt_us=dt_us,
            )
            front_confidence = self._echo_confidence(
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
            back_time_us = self._refine_peak_time_us(envelope_smooth, time_s, int(back_wall_idx))
            back_std_us = self._estimate_time_std_us(
                envelope_smooth=envelope_smooth,
                peak_idx=int(back_wall_idx),
                prominence=prominence_by_idx.get(int(back_wall_idx), 0.0),
                peak_height=peak_height,
                noise_sigma=noise_sigma,
                dt_us=dt_us,
            )
            back_confidence = self._echo_confidence(
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

    def resolve_defects(
        self,
        sample: NdtSampleRecord,
        signal_analysis: NdtSignalAnalysisRecord | None = None,
    ) -> list[NdtAnalyzedDefect]:
        """Return fused defect list combining metadata and signal-derived detections."""
        analysis = signal_analysis or self.analyze_signal(sample)
        signal_candidates = self._detect_from_signal(sample, analysis)
        return self._merge_metadata_and_signal(sample, signal_candidates)

    def _detect_from_signal(
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
        ) = self._detect_peaks(sample, envelope)
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
                    time_us=self._refine_peak_time_us(envelope_smooth, time_s, peak_idx_int),
                    confidence=confidence,
                    source="signal",
                )
            )

        return self._deduplicate_by_depth(candidates, tolerance_m=0.00045)

    def _detect_peaks(
        self,
        sample: NdtSampleRecord,
        envelope: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
        fs_hz = float(sample.fs_hz)

        smooth_window = max(3, int(round(0.10e-6 * fs_hz)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        envelope_smooth = np.convolve(envelope, kernel, mode="same")

        noise_region_len = max(32, int(0.15 * envelope_smooth.size))
        noise_region = envelope_smooth[:noise_region_len]
        noise_median = float(np.median(noise_region))
        noise_mad = float(np.median(np.abs(noise_region - noise_median)))
        noise_sigma = max(1e-12, 1.4826 * noise_mad)

        relative_height = self.min_relative_height * float(np.max(envelope_smooth))
        absolute_height = noise_median + self.noise_sigma_factor * noise_sigma
        peak_height = max(relative_height, absolute_height)

        min_peak_distance = max(8, int(round(self.peak_distance_us * 1e-6 * fs_hz)))
        peak_prominence = max(
            3.0 * noise_sigma,
            0.04 * float(np.max(envelope_smooth)),
        )

        peak_indices, properties = find_peaks(
            envelope_smooth,
            height=peak_height,
            distance=min_peak_distance,
            prominence=peak_prominence,
        )
        peak_prominences = np.asarray(
            properties.get("prominences", np.zeros(peak_indices.size, dtype=np.float64)),
            dtype=np.float64,
        )
        if peak_prominences.size != peak_indices.size:
            peak_prominences = np.zeros(peak_indices.size, dtype=np.float64)

        return (
            envelope_smooth,
            peak_indices,
            peak_prominences,
            peak_height,
            noise_sigma,
            min_peak_distance,
        )

    def _find_front_wall(self, peak_indices: np.ndarray, time_s: np.ndarray) -> int | None:
        start_time_s = 0.08e-6
        for peak_idx in peak_indices:
            if time_s[int(peak_idx)] >= start_time_s:
                return int(peak_idx)
        return None

    def _find_back_wall(
        self,
        sample: NdtSampleRecord,
        peak_indices: np.ndarray,
        time_s: np.ndarray,
        envelope_smooth: np.ndarray,
        front_wall_idx: int,
    ) -> int | None:
        front_time_s = float(time_s[front_wall_idx])
        candidate_indices = [
            int(p) for p in peak_indices if time_s[int(p)] > front_time_s + 0.45e-6
        ]
        if not candidate_indices:
            return None

        if sample.thickness_m is not None and sample.c_mps > 0:
            expected_back_time_s = front_time_s + 2.0 * float(sample.thickness_m) / float(
                sample.c_mps
            )
            tolerance_s = max(0.8e-6, 0.20 * (expected_back_time_s - front_time_s))
            nearest_idx = min(
                candidate_indices, key=lambda p: abs(float(time_s[p]) - expected_back_time_s)
            )
            if abs(float(time_s[nearest_idx]) - expected_back_time_s) <= tolerance_s:
                return nearest_idx

        return max(candidate_indices, key=lambda p: float(envelope_smooth[p]))

    def _estimate_depth(
        self,
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

    def _sample_period_us(self, time_s: np.ndarray) -> float:
        if time_s.size < 2:
            return 0.02
        dt = float(np.median(np.diff(time_s)))
        return max(1e-6, dt * 1e6)

    def _refine_peak_time_us(
        self, envelope: np.ndarray, time_s: np.ndarray, peak_idx: int
    ) -> float:
        base_time_us = float(time_s[peak_idx] * 1e6)
        if peak_idx <= 0 or peak_idx >= envelope.size - 1:
            return base_time_us

        y_prev = float(envelope[peak_idx - 1])
        y_mid = float(envelope[peak_idx])
        y_next = float(envelope[peak_idx + 1])
        denom = y_prev - 2.0 * y_mid + y_next
        if abs(denom) < 1e-12:
            return base_time_us

        offset = 0.5 * (y_prev - y_next) / denom
        offset = float(np.clip(offset, -1.0, 1.0))
        return base_time_us + offset * self._sample_period_us(time_s)

    def _estimate_time_std_us(
        self,
        envelope_smooth: np.ndarray,
        peak_idx: int,
        prominence: float,
        peak_height: float,
        noise_sigma: float,
        dt_us: float,
    ) -> float:
        try:
            widths = peak_widths(envelope_smooth, np.array([peak_idx]), rel_height=0.5)[0]
            width_samples = float(widths[0]) if widths.size else 1.0
        except Exception:
            width_samples = 1.0

        amplitude = float(envelope_smooth[peak_idx])
        snr = amplitude / max(noise_sigma, 1e-12)
        snr_gain = max(1.0, np.sqrt(max(snr, 1.0)))
        prominence_gain = max(1.0, np.sqrt(max(prominence / max(peak_height, 1e-12), 1.0)))
        std_us = (width_samples * dt_us) / (2.355 * snr_gain * prominence_gain)
        return float(max(0.5 * dt_us, min(std_us, 10.0 * dt_us)))

    def _echo_confidence(
        self,
        amplitude: float,
        prominence: float,
        peak_height: float,
        noise_sigma: float,
    ) -> float:
        snr = amplitude / max(noise_sigma, 1e-12)
        snr_score = float(np.clip((snr - 1.0) / 12.0, 0.0, 1.0))
        prominence_score = float(np.clip(prominence / max(peak_height, 1e-12), 0.0, 1.0))
        return float(np.clip(0.65 * snr_score + 0.35 * prominence_score, 0.0, 1.0))

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

    def _compute_thinning_flag(
        self,
        sample: NdtSampleRecord,
        estimated_thickness_mm: float | None,
    ) -> bool:
        if estimated_thickness_mm is None:
            return False

        context_text = f"{sample.name} {sample.description}"
        thinning_context = bool(re.search(r"corrosion|thinning|thinned|corrod", context_text, re.I))
        reference_match = re.search(r"original\s*([0-9]+(?:\.[0-9]+)?)\s*mm", context_text, re.I)

        if reference_match:
            original_mm = float(reference_match.group(1))
            return estimated_thickness_mm < 0.95 * original_mm

        if thinning_context:
            return True

        return False

    def _merge_metadata_and_signal(
        self,
        sample: NdtSampleRecord,
        signal_candidates: Sequence[NdtAnalyzedDefect],
    ) -> list[NdtAnalyzedDefect]:
        metadata_candidates = [
            NdtAnalyzedDefect(
                depth_m=float(defect.depth_m),
                amplitude=defect.amplitude,
                time_us=(
                    float((2.0 * float(defect.depth_m) / float(sample.c_mps)) * 1e6)
                    if sample.c_mps > 0.0
                    else None
                ),
                confidence=0.92 if defect.amplitude is not None else 0.82,
                source="metadata",
            )
            for defect in sample.defects
            if defect.depth_m is not None and defect.depth_m >= 0.0
        ]

        if not metadata_candidates:
            filtered_signal = [
                candidate for candidate in signal_candidates if candidate.confidence >= 0.40
            ]
            return self._deduplicate_by_depth(filtered_signal, tolerance_m=0.00045)
        if not signal_candidates:
            return self._deduplicate_by_depth(metadata_candidates, tolerance_m=0.00045)

        remaining_signal = list(signal_candidates)
        fused: list[NdtAnalyzedDefect] = []
        merge_tolerance_m = 0.00100

        for metadata in metadata_candidates:
            best_idx = -1
            best_distance = float("inf")
            for idx, signal in enumerate(remaining_signal):
                distance = abs(float(signal.depth_m) - float(metadata.depth_m))
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            if best_idx >= 0 and best_distance <= merge_tolerance_m:
                signal = remaining_signal.pop(best_idx)
                fused.append(
                    NdtAnalyzedDefect(
                        depth_m=float(
                            (0.65 * float(metadata.depth_m)) + (0.35 * float(signal.depth_m))
                        ),
                        amplitude=(
                            metadata.amplitude
                            if metadata.amplitude is not None
                            else signal.amplitude
                        ),
                        time_us=signal.time_us if signal.time_us is not None else metadata.time_us,
                        confidence=float(
                            min(
                                1.0,
                                max(float(metadata.confidence), float(signal.confidence) + 0.15),
                            )
                        ),
                        source="fused",
                    )
                )
            else:
                fused.append(metadata)

        for signal in remaining_signal:
            if signal.confidence >= 0.45:
                fused.append(signal)

        return self._deduplicate_by_depth(fused, tolerance_m=0.00045)

    def _deduplicate_by_depth(
        self, candidates: Sequence[NdtAnalyzedDefect], tolerance_m: float
    ) -> list[NdtAnalyzedDefect]:
        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda item: (
                float(item.depth_m),
                -float(item.confidence),
            ),
        )

        merged: list[NdtAnalyzedDefect] = []
        for candidate in ordered:
            if not merged:
                merged.append(candidate)
                continue

            if abs(float(candidate.depth_m) - float(merged[-1].depth_m)) <= tolerance_m:
                if candidate.confidence > merged[-1].confidence:
                    merged[-1] = candidate
            else:
                merged.append(candidate)

        return merged
