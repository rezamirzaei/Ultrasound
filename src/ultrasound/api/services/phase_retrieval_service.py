"""Service layer for real-ultrasound phase retrieval previews."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from ultrasound.api.config import AppConfig
from ultrasound.api.models.schemas import (
    PhaseRetrievalPreviewRequest,
    PhaseRetrievalPreviewResponse,
    PhaseRetrievalStatusResponse,
)
from ultrasound.api.services.service_errors import DependencyUnavailableError, InvalidRequestError
from ultrasound.data.picmus_dataset import (
    PICMUS_SOURCE_URL,
    default_picmus_case,
    list_picmus_rf_cases,
    picmus_in_vivo_available,
    resolve_picmus_in_vivo_root,
)
from ultrasound.workflows import run_phase_retrieval_picmus


class PhaseRetrievalService:
    """Expose tuned PICMUS-backed phase retrieval to the API/UI layer."""

    DEFAULT_SEGMENT_LENGTH = 96
    DEFAULT_MEASUREMENT_RATIO = 5
    DEFAULT_MAX_ITERATIONS = 150
    DEFAULT_SOLVER: Literal["lbfgs", "wirtinger"] = "lbfgs"

    def __init__(self, config: AppConfig):
        self.config = config
        self._root_dir = self.config.data_dir / "picmus"

    def _dataset_hint(self) -> str:
        return (
            "PICMUS in-vivo RF data is not available locally. "
            "Run `python scripts/download_picmus_in_vivo.py` first."
        )

    def get_status(self) -> PhaseRetrievalStatusResponse:
        available = picmus_in_vivo_available(self._root_dir)
        cases = list_picmus_rf_cases(self._root_dir) if available else []
        recommended_case = default_picmus_case(self._root_dir)
        return PhaseRetrievalStatusResponse(
            generated_at=datetime.now(tz=timezone.utc),
            dataset_available=available,
            dataset_root=str(resolve_picmus_in_vivo_root(self._root_dir)),
            source_url=PICMUS_SOURCE_URL,
            available_cases=cases,
            recommended_case=recommended_case,
            recommended_segment_length=self.DEFAULT_SEGMENT_LENGTH,
            recommended_measurement_ratio=self.DEFAULT_MEASUREMENT_RATIO,
            recommended_solver=self.DEFAULT_SOLVER,
        )

    def preview(self, request: PhaseRetrievalPreviewRequest) -> PhaseRetrievalPreviewResponse:
        if not picmus_in_vivo_available(self._root_dir):
            raise DependencyUnavailableError(self._dataset_hint())

        available_cases = set(list_picmus_rf_cases(self._root_dir))
        if request.case_name not in available_cases:
            raise InvalidRequestError(
                f"Unknown PICMUS case {request.case_name!r}. Available cases: {sorted(available_cases)}"
            )

        try:
            result = run_phase_retrieval_picmus(
                root_dir=str(self._root_dir),
                case_name=request.case_name,
                segment_length=request.segment_length,
                measurement_ratio=request.measurement_ratio,
                n_iter=request.max_iterations,
                seed=request.seed,
                solver=self.DEFAULT_SOLVER,
                angle_index=request.angle_index,
                element_index=request.element_index,
                start_index=request.start_index,
            )
        except FileNotFoundError as exc:
            raise DependencyUnavailableError(self._dataset_hint()) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

        metadata = result.signal_metadata or {}
        report = result.report
        solver = "wirtinger" if report.get("solver") == "wirtinger" else self.DEFAULT_SOLVER
        return PhaseRetrievalPreviewResponse(
            generated_at=datetime.now(tz=timezone.utc),
            dataset_name=str(metadata.get("dataset", "PICMUS in_vivo")),
            case_name=str(metadata.get("case_name", request.case_name)),
            angle_index=int(metadata.get("angle_index", 0)),
            element_index=int(metadata.get("element_index", 0)),
            start_index=int(metadata.get("start_index", 0)),
            segment_length=int(metadata.get("segment_length", request.segment_length)),
            energy=float(metadata.get("energy", 0.0)),
            sampling_frequency_hz=float(metadata.get("sampling_frequency_hz", 20e6)),
            sound_speed_mps=float(metadata.get("sound_speed_mps", 1540.0)),
            solver=solver,
            measurement_ratio=int(report.get("measurement_ratio", request.measurement_ratio)),
            measurement_count=int(report.get("m", 1)),
            optimization_iterations=int(report.get("optimization_iterations", 0)),
            optimization_success=bool(report.get("optimization_success", True)),
            init_relative_error=float(report.get("init_relative_error", 0.0)),
            final_relative_error=float(report.get("final_relative_error", 0.0)),
            error_reduced=bool(result.status.get("error_reduced", False)),
            overall_pass=bool(result.status.get("overall_pass", False)),
            true_real=[float(v) for v in result.x_true.real],
            recovered_real=[float(v) for v in result.x_aligned.real],
            true_imag=[float(v) for v in result.x_true.imag],
            recovered_imag=[float(v) for v in result.x_aligned.imag],
            amplitude_rmse_curve=[float(v) for v in result.amplitude_rmse],
        )
