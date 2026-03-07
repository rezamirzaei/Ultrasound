"""Service layer for real transcranial ultrasound phase-retrieval previews."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, cast

import numpy as np

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    PhaseRetrievalPreviewRequest,
    PhaseRetrievalPreviewResponse,
    PhaseRetrievalStatusResponse,
)
from ultrasound.api.services.interfaces import MediaRenderer
from ultrasound.api.services.service_errors import DependencyUnavailableError, InvalidRequestError
from ultrasound.data.transcranial_phase_dataset import (
    ETH_TRANSCRANIAL_SOURCE_URL,
    default_transcranial_case,
    list_transcranial_scan_cases,
    resolve_transcranial_dataset_root,
    transcranial_dataset_available,
)
from ultrasound.workflows import run_phase_retrieval_transcranial


class PhaseRetrievalService:
    """Expose transcranial hydrophone phase retrieval to the API/UI layer."""

    DEFAULT_WINDOW_LENGTH = 256
    DEFAULT_N_FFT = 80
    DEFAULT_HOP_LENGTH = 8
    DEFAULT_MAX_ITERATIONS = 120
    DEFAULT_SOLVER: Literal["griffin_lim"] = "griffin_lim"

    def __init__(self, config: AppConfig, media_service: MediaRenderer):
        self.config = config
        self.media_service = media_service
        self._root_dir = self.config.data_dir / "phase_retrieval"

    def _dataset_hint(self) -> str:
        return (
            "ETH transcranial hydrophone scan data is not available locally. "
            "Run `python scripts/download_transcranial_phase_retrieval.py` first."
        )

    @staticmethod
    def _normalize_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=np.float64)
        array = array - float(np.min(array))
        peak = float(np.max(array))
        if peak <= 1e-12:
            return np.zeros(array.shape, dtype=np.uint8)
        return np.clip(255.0 * array / peak, 0.0, 255.0).astype(np.uint8)

    def _energy_map_data_url(self, energy_map: np.ndarray, *, row_index: int, col_index: int) -> str:
        base = self._normalize_uint8(np.sqrt(np.maximum(energy_map, 0.0)))
        rgb = np.repeat(base[:, :, None], 3, axis=2)
        radius = max(2, min(rgb.shape[0], rgb.shape[1]) // 24)
        row_start = max(0, row_index - radius)
        row_end = min(rgb.shape[0], row_index + radius + 1)
        col_start = max(0, col_index - radius)
        col_end = min(rgb.shape[1], col_index + radius + 1)
        rgb[row_start:row_end, col_index, :] = np.array([255, 64, 64], dtype=np.uint8)
        rgb[row_index, col_start:col_end, :] = np.array([255, 64, 64], dtype=np.uint8)
        return self.media_service.as_png_data_url(rgb)

    def _spectrogram_data_url(self, spectrogram: np.ndarray) -> str:
        image = self._normalize_uint8(np.log1p(np.maximum(spectrogram, 0.0)))[::-1, :]
        return self.media_service.as_png_data_url(image)

    def get_status(self) -> PhaseRetrievalStatusResponse:
        available = transcranial_dataset_available(self._root_dir)
        cases = list_transcranial_scan_cases(self._root_dir) if available else []
        recommended_case = default_transcranial_case(self._root_dir)
        return PhaseRetrievalStatusResponse(
            generated_at=datetime.now(tz=timezone.utc),
            dataset_available=available,
            dataset_root=str(resolve_transcranial_dataset_root(self._root_dir)),
            source_url=ETH_TRANSCRANIAL_SOURCE_URL,
            available_cases=cases,
            recommended_case=recommended_case,
            recommended_window_length=self.DEFAULT_WINDOW_LENGTH,
            recommended_n_fft=self.DEFAULT_N_FFT,
            recommended_hop_length=self.DEFAULT_HOP_LENGTH,
            recommended_solver=self.DEFAULT_SOLVER,
            recovered_quantity=(
                "Recovered output is the time-domain hydrophone pulse and its missing phase, "
                "using only STFT magnitude measurements."
            ),
            measurement_description=(
                "Input measurement is the magnitude of the short-time Fourier transform of a "
                "real hydrophone waveform selected from an ETH transcranial scan plane."
            ),
        )

    def preview(self, request: PhaseRetrievalPreviewRequest) -> PhaseRetrievalPreviewResponse:
        if not transcranial_dataset_available(self._root_dir):
            raise DependencyUnavailableError(self._dataset_hint())

        available_cases = set(list_transcranial_scan_cases(self._root_dir))
        if request.case_name not in available_cases:
            raise InvalidRequestError(
                "Unknown transcranial scan case "
                f"{request.case_name!r}. Available cases: {sorted(available_cases)}"
            )
        if request.hop_length >= request.n_fft:
            raise InvalidRequestError("hop_length must be smaller than n_fft")

        try:
            result = run_phase_retrieval_transcranial(
                root_dir=str(self._root_dir),
                case_name=request.case_name,
                window_length=request.window_length,
                n_fft=request.n_fft,
                hop_length=request.hop_length,
                n_iter=request.max_iterations,
                seed=request.seed,
                row_index=request.row_index,
                col_index=request.col_index,
                start_index=request.start_index,
            )
        except FileNotFoundError as exc:
            raise DependencyUnavailableError(self._dataset_hint()) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

        metadata = result.signal_metadata or {}
        report = result.report
        row_index = int(metadata.get("row_index", 0))
        col_index = int(metadata.get("col_index", 0))
        plane = str(metadata.get("plane", "XY"))
        if plane not in {"XY", "XZ"}:
            plane = "XY"
        scan_map = result.scan_energy_map if result.scan_energy_map is not None else np.zeros((8, 8))
        return PhaseRetrievalPreviewResponse(
            generated_at=datetime.now(tz=timezone.utc),
            dataset_name=str(metadata.get("dataset", "ETH transcranial hydrophone scans")),
            case_name=str(metadata.get("case_name", request.case_name)),
            plane=cast(Literal["XY", "XZ"], plane),
            row_index=row_index,
            col_index=col_index,
            start_index=int(metadata.get("start_index", 0)),
            window_length=int(metadata.get("window_length", request.window_length)),
            trace_energy=float(metadata.get("trace_energy", 0.0)),
            dominant_frequency_bin=int(metadata.get("dominant_frequency_bin", 1)),
            solver="griffin_lim",
            n_fft=int(report.get("n_fft", request.n_fft)),
            hop_length=int(report.get("hop_length", request.hop_length)),
            optimization_iterations=int(report.get("optimization_iterations", request.max_iterations)),
            init_relative_error=float(report.get("init_relative_error", 0.0)),
            final_relative_error=float(report.get("final_relative_error", 0.0)),
            signal_correlation=float(report.get("signal_correlation", 0.0)),
            phase_rmse=float(report.get("phase_rmse", 0.0)),
            initial_consistency_error=float(report.get("initial_consistency_error", 0.0)),
            final_consistency_error=float(report.get("final_consistency_error", 0.0)),
            error_reduced=bool(result.status.get("error_reduced", False)),
            overall_pass=bool(result.status.get("overall_pass", False)),
            true_signal=[float(v) for v in result.true_signal],
            recovered_signal=[float(v) for v in result.recovered_signal],
            true_phase_spectrum=[float(v) for v in result.true_phase_spectrum],
            recovered_phase_spectrum=[float(v) for v in result.recovered_phase_spectrum],
            residual_curve=[float(v) for v in result.residual_curve],
            scan_image_data_url=self._energy_map_data_url(
                scan_map,
                row_index=row_index,
                col_index=col_index,
            ),
            spectrogram_image_data_url=self._spectrogram_data_url(result.measured_spectrogram),
        )
