"""Fusion utilities for metadata and signal-derived NDT defect candidates."""

from __future__ import annotations

from collections.abc import Sequence

from ultrasound.api.models.domain import NdtAnalyzedDefect, NdtSampleRecord


class NdtDefectFusionAnalyzer:
    """Merge metadata defects with signal-derived candidates."""

    def merge(
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
            return self.deduplicate_by_depth(filtered_signal, tolerance_m=0.00045)
        if not signal_candidates:
            return self.deduplicate_by_depth(metadata_candidates, tolerance_m=0.00045)

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
                        depth_m=float((0.65 * float(metadata.depth_m)) + (0.35 * float(signal.depth_m))),
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

        return self.deduplicate_by_depth(fused, tolerance_m=0.00045)

    def deduplicate_by_depth(
        self,
        candidates: Sequence[NdtAnalyzedDefect],
        tolerance_m: float,
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
