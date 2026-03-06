"""Shared helpers for YOLO lab prediction flows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ultrasound.api.models.schemas import YoloPredictRequest
from ultrasound.api.services.service_errors import InvalidRequestError

_GENERIC_REQUESTED_MODEL = ""


def request_matches_model_path(requested_model: str, model_path: Path) -> bool:
    requested = requested_model.strip()
    if not requested:
        return False
    requested_path = Path(requested).expanduser()
    try:
        return requested_path.resolve(strict=False) == model_path.resolve(strict=False)
    except OSError:
        return requested_path == model_path


def prefer_existing_model(
    request: YoloPredictRequest,
    *,
    default_model_candidates: Sequence[str],
    preferred_model_path: Path | None,
    missing_explicit_preferred_message: str | None = None,
) -> YoloPredictRequest:
    if preferred_model_path is None:
        return request

    requested_model = (request.model or "").strip()
    if (
        missing_explicit_preferred_message is not None
        and request_matches_model_path(requested_model, preferred_model_path)
        and not preferred_model_path.exists()
    ):
        raise InvalidRequestError(missing_explicit_preferred_message)

    generic_models = {_GENERIC_REQUESTED_MODEL, *default_model_candidates}
    if requested_model in generic_models and preferred_model_path.exists():
        return request.model_copy(update={"model": str(preferred_model_path)})
    return request
