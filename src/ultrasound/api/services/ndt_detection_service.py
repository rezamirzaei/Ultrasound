"""NDT waveform-based defect detection and metadata fusion."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import find_peaks, hilbert

from ultrasound.api.models.domain import NdtAnalyzedDefect, NdtSampleRecord


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

    def resolve_defects(self, sample: NdtSampleRecord) -> list[NdtAnalyzedDefect]:
        """Return fused defect list combining metadata and signal-derived detections."""
        signal_candidates = self._detect_from_signal(sample)
        return self._merge_metadata_and_signal(sample, signal_candidates)

    def _detect_from_signal(self, sample: NdtSampleRecord) -> list[NdtAnalyzedDefect]:
        rf = np.asarray(sample.rf, dtype=np.float64).reshape(-1)
        time_s = np.asarray(sample.time, dtype=np.float64).reshape(-1)
        if rf.size < 16:
            return []

        fs_hz = float(sample.fs_hz)
        c_mps = float(sample.c_mps)

        envelope = np.abs(hilbert(rf))
        smooth_window = max(3, int(round(0.10e-6 * fs_hz)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        envelope_smooth = np.convolve(envelope, kernel, mode="same")

        noise_region_len = max(32, int(0.15 * envelope_smooth.size))
        noise_region = envelope_smooth[:noise_region_len]
        noise_median = float(np.median(noise_region))
        noise_mad = float(np.median(np.abs(noise_region - noise_median)))
        noise_sigma = 1.4826 * noise_mad

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
        if peak_indices.size == 0:
            return []

        front_wall_idx = self._find_front_wall(peak_indices, time_s)
        if front_wall_idx is None:
            return []

        back_wall_idx = self._find_back_wall(
            sample=sample,
            peak_indices=peak_indices,
            time_s=time_s,
            envelope_smooth=envelope_smooth,
            front_wall_idx=front_wall_idx,
        )

        peak_prominences = np.asarray(
            properties.get("prominences", np.zeros(peak_indices.size, dtype=np.float64)),
            dtype=np.float64,
        )
        if peak_prominences.size != peak_indices.size:
            peak_prominences = np.zeros(peak_indices.size, dtype=np.float64)

        max_envelope = max(float(np.max(envelope_smooth)), 1e-12)
        prominence_scale = max(peak_height, 1e-12)
        gate_distance = max(min_peak_distance, int(round(0.25e-6 * fs_hz)))

        candidates: list[NdtAnalyzedDefect] = []
        for idx, peak_idx in enumerate(peak_indices):
            if peak_idx <= front_wall_idx:
                continue
            if abs(int(peak_idx) - int(front_wall_idx)) <= gate_distance:
                continue
            if back_wall_idx is not None and peak_idx >= back_wall_idx:
                continue
            if (
                back_wall_idx is not None
                and abs(int(peak_idx) - int(back_wall_idx)) <= gate_distance
            ):
                continue

            depth_m = self._estimate_depth(
                time_s=time_s,
                peak_idx=int(peak_idx),
                front_wall_idx=int(front_wall_idx),
                back_wall_idx=int(back_wall_idx) if back_wall_idx is not None else None,
                c_mps=c_mps,
                thickness_m=sample.thickness_m,
            )
            if depth_m is None or depth_m <= 0.0:
                continue
            if sample.thickness_m is not None and depth_m >= 0.98 * float(sample.thickness_m):
                continue

            amplitude = float(envelope_smooth[int(peak_idx)])
            amp_norm = float(np.clip(amplitude / max_envelope, 0.0, 1.0))
            prominence_norm = float(np.clip(peak_prominences[idx] / prominence_scale, 0.0, 1.0))
            confidence = float(np.clip(0.45 * amp_norm + 0.55 * prominence_norm, 0.0, 1.0))
            if confidence < 0.25:
                continue

            candidates.append(
                NdtAnalyzedDefect(
                    depth_m=float(depth_m),
                    amplitude=amplitude,
                    time_us=float(time_s[int(peak_idx)] * 1e6),
                    confidence=confidence,
                    source="signal",
                )
            )

        return self._deduplicate_by_depth(candidates, tolerance_m=0.00045)

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
            tolerance_s = max(0.7e-6, 0.18 * (expected_back_time_s - front_time_s))
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
