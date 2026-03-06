"""Peak and wall-echo detection utilities for NDT A-scan analysis."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from ultrasound.api.models.domain import NdtSampleRecord


class NdtPeakDetector:
    """Detect salient envelope peaks and derive wall-echo timing metadata."""

    def __init__(
        self,
        min_relative_height: float = 0.22,
        noise_sigma_factor: float = 5.0,
        peak_distance_us: float = 0.35,
    ) -> None:
        self.min_relative_height = float(min_relative_height)
        self.noise_sigma_factor = float(noise_sigma_factor)
        self.peak_distance_us = float(peak_distance_us)

    def detect_peaks(
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
        peak_prominence = max(3.0 * noise_sigma, 0.04 * float(np.max(envelope_smooth)))

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

    @staticmethod
    def find_front_wall(peak_indices: np.ndarray, time_s: np.ndarray) -> int | None:
        start_time_s = 0.08e-6
        for peak_idx in peak_indices:
            if time_s[int(peak_idx)] >= start_time_s:
                return int(peak_idx)
        return None

    def find_back_wall(
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
                candidate_indices,
                key=lambda p: abs(float(time_s[p]) - expected_back_time_s),
            )
            if abs(float(time_s[nearest_idx]) - expected_back_time_s) <= tolerance_s:
                return nearest_idx

        return max(candidate_indices, key=lambda p: float(envelope_smooth[p]))

    @staticmethod
    def sample_period_us(time_s: np.ndarray) -> float:
        if time_s.size < 2:
            return 0.02
        dt = float(np.median(np.diff(time_s)))
        return max(1e-6, dt * 1e6)

    def refine_peak_time_us(self, envelope: np.ndarray, time_s: np.ndarray, peak_idx: int) -> float:
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
        return base_time_us + offset * self.sample_period_us(time_s)

    def estimate_time_std_us(
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

    @staticmethod
    def echo_confidence(
        amplitude: float,
        prominence: float,
        peak_height: float,
        noise_sigma: float,
    ) -> float:
        snr = amplitude / max(noise_sigma, 1e-12)
        snr_score = float(np.clip((snr - 1.0) / 12.0, 0.0, 1.0))
        prominence_score = float(np.clip(prominence / max(peak_height, 1e-12), 0.0, 1.0))
        return float(np.clip(0.65 * snr_score + 0.35 * prominence_score, 0.0, 1.0))
