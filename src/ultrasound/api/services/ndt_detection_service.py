"""NDT waveform defect analysis facade built from smaller analyzers."""

from __future__ import annotations

from ultrasound.api.models.domain import NdtAnalyzedDefect, NdtSampleRecord, NdtSignalAnalysisRecord
from ultrasound.api.services.ndt_defect_fusion_analyzer import NdtDefectFusionAnalyzer
from ultrasound.api.services.ndt_peak_detector import NdtPeakDetector
from ultrasound.api.services.ndt_signal_defect_analyzer import NdtSignalDefectAnalyzer
from ultrasound.api.services.ndt_thickness_analyzer import NdtThicknessAnalyzer


class NdtDetectionService:
    """Coordinate signal analysis, defect proposal, and metadata fusion."""

    def __init__(
        self,
        min_relative_height: float = 0.22,
        noise_sigma_factor: float = 5.0,
        peak_distance_us: float = 0.35,
    ) -> None:
        peak_detector = NdtPeakDetector(
            min_relative_height=min_relative_height,
            noise_sigma_factor=noise_sigma_factor,
            peak_distance_us=peak_distance_us,
        )
        self.thickness_analyzer = NdtThicknessAnalyzer(peak_detector)
        self.signal_defect_analyzer = NdtSignalDefectAnalyzer(peak_detector)
        self.defect_fusion_analyzer = NdtDefectFusionAnalyzer()

    def analyze_signal(self, sample: NdtSampleRecord) -> NdtSignalAnalysisRecord:
        return self.thickness_analyzer.analyze_signal(sample)

    def resolve_defects(
        self,
        sample: NdtSampleRecord,
        signal_analysis: NdtSignalAnalysisRecord | None = None,
    ) -> list[NdtAnalyzedDefect]:
        analysis = signal_analysis or self.analyze_signal(sample)
        signal_candidates = self.signal_defect_analyzer.detect(sample, analysis)
        return self.defect_fusion_analyzer.merge(sample, signal_candidates)
