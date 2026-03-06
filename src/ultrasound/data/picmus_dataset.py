"""Helpers for loading and downloading PICMUS in-vivo raw ultrasound RF data."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

PICMUS_IN_VIVO_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/"
    "sites/www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/in_vivo.zip"
)
PICMUS_SOURCE_URL = "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home"
_DEFAULT_CASES = ("carotid_cross", "carotid_long")


@dataclass(frozen=True)
class PicmusRfSegment:
    """One deterministic PICMUS RF segment selected for downstream workflows."""

    case_name: str
    angle_index: int
    element_index: int
    start_index: int
    segment_length: int
    energy: float
    rf_segment: np.ndarray
    sampling_frequency_hz: float
    sound_speed_mps: float
    probe_geometry: np.ndarray


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in runtime only
        raise ImportError(
            "PICMUS support requires h5py. Install the project dependencies before loading "
            "PICMUS HDF5 data."
        ) from exc
    return h5py


def resolve_picmus_in_vivo_root(root_dir: str | Path | None = None) -> Path:
    """Resolve the extracted PICMUS in-vivo dataset directory."""
    base = Path(root_dir or Path("data") / "picmus").expanduser().resolve()
    candidates = (
        base,
        base / "in_vivo",
        base / "in_vivo" / "in_vivo",
    )
    for candidate in candidates:
        if any(candidate.glob("*/*_dataset_rf.hdf5")):
            return candidate
    return base / "in_vivo"


def list_picmus_rf_cases(root_dir: str | Path | None = None) -> list[str]:
    """Return PICMUS case names with RF HDF5 data available locally."""
    root = resolve_picmus_in_vivo_root(root_dir)
    cases = sorted(path.parent.name for path in root.glob("*/*_dataset_rf.hdf5"))
    return cases


def picmus_in_vivo_available(root_dir: str | Path | None = None) -> bool:
    """Return whether RF data for the PICMUS in-vivo set is present locally."""
    return bool(list_picmus_rf_cases(root_dir))


def _case_rf_path(root_dir: str | Path | None, case_name: str) -> Path:
    root = resolve_picmus_in_vivo_root(root_dir)
    case_path = root / case_name / f"{case_name}_expe_dataset_rf.hdf5"
    if not case_path.exists():
        available = list_picmus_rf_cases(root)
        raise FileNotFoundError(
            f"PICMUS RF case not found: {case_path}. Available cases: {available}"
        )
    return case_path


def _read_scalar(dataset: Any, default: float) -> float:
    try:
        return float(np.asarray(dataset).reshape(-1)[0])
    except Exception:
        return float(default)


def _load_trace(h5_file: Any, angle_index: int, element_index: int) -> np.ndarray:
    data = h5_file["US"]["US_DATASET0000"]["data"]
    real = np.asarray(data["real"][angle_index, element_index, :], dtype=np.float64)
    imag_ds = data.get("imag")
    if imag_ds is None:
        return real

    imag = np.asarray(imag_ds[angle_index, element_index, :], dtype=np.float64)
    if np.max(np.abs(imag)) <= 1e-12:
        return real
    return real + 1j * imag


def load_picmus_rf_segment(
    root_dir: str | Path | None,
    *,
    case_name: str,
    angle_index: int,
    element_index: int,
    start_index: int,
    segment_length: int,
) -> PicmusRfSegment:
    """Load one explicitly indexed PICMUS RF segment."""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if angle_index < 0 or element_index < 0 or start_index < 0:
        raise ValueError("angle_index, element_index, and start_index must be non-negative")

    h5py = _require_h5py()
    case_path = _case_rf_path(root_dir, case_name)
    with h5py.File(case_path, "r") as h5_file:
        data = h5_file["US"]["US_DATASET0000"]["data"]
        n_angles, n_elements, n_samples = data["real"].shape
        if angle_index >= n_angles:
            raise ValueError(f"angle_index must be < {n_angles}")
        if element_index >= n_elements:
            raise ValueError(f"element_index must be < {n_elements}")
        if start_index + segment_length > n_samples:
            raise ValueError(
                f"Requested segment exceeds trace length {n_samples}: start={start_index}, "
                f"length={segment_length}"
            )

        trace = _load_trace(h5_file, angle_index, element_index)
        segment = np.asarray(
            np.real(trace[start_index : start_index + segment_length]),
            dtype=np.float64,
        )
        meta = h5_file["US"]["US_DATASET0000"]
        return PicmusRfSegment(
            case_name=case_name,
            angle_index=int(angle_index),
            element_index=int(element_index),
            start_index=int(start_index),
            segment_length=int(segment_length),
            energy=float(np.sum(np.abs(segment) ** 2)),
            rf_segment=segment,
            sampling_frequency_hz=_read_scalar(meta.get("sampling_frequency"), 20e6),
            sound_speed_mps=_read_scalar(meta.get("sound_speed"), 1540.0),
            probe_geometry=np.asarray(meta.get("probe_geometry", np.empty((0, 3))), dtype=np.float64),
        )


def select_high_energy_rf_segment(
    root_dir: str | Path | None,
    *,
    case_name: str,
    segment_length: int = 96,
    angle_index: int | None = None,
    element_index: int | None = None,
) -> PicmusRfSegment:
    """Select the highest-energy RF window for one PICMUS case."""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")

    h5py = _require_h5py()
    case_path = _case_rf_path(root_dir, case_name)
    with h5py.File(case_path, "r") as h5_file:
        real = h5_file["US"]["US_DATASET0000"]["data"]["real"]
        n_angles, n_elements, n_samples = real.shape
        if segment_length > n_samples:
            raise ValueError(
                f"segment_length must be <= trace length {n_samples}, got {segment_length}"
            )

        angle_indices = [angle_index] if angle_index is not None else list(range(n_angles))
        element_indices = [element_index] if element_index is not None else list(range(n_elements))

        best_score = -float("inf")
        best_window: tuple[int, int, int] | None = None

        for cur_angle in angle_indices:
            if cur_angle is None or cur_angle < 0 or cur_angle >= n_angles:
                raise ValueError(f"angle_index must be between 0 and {n_angles - 1}")
            for cur_element in element_indices:
                if cur_element is None or cur_element < 0 or cur_element >= n_elements:
                    raise ValueError(f"element_index must be between 0 and {n_elements - 1}")
                trace = np.asarray(real[cur_angle, cur_element, :], dtype=np.float64)
                window_energy = np.convolve(trace**2, np.ones(segment_length), mode="valid")
                start_index = int(np.argmax(window_energy))
                score = float(window_energy[start_index] / segment_length)
                if score > best_score:
                    best_score = score
                    best_window = (int(cur_angle), int(cur_element), start_index)

        if best_window is None:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Failed to select a PICMUS RF segment for case {case_name}")

        return load_picmus_rf_segment(
            root_dir,
            case_name=case_name,
            angle_index=best_window[0],
            element_index=best_window[1],
            start_index=best_window[2],
            segment_length=segment_length,
        )


def download_picmus_in_vivo(
    dest_dir: str | Path,
    *,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download and extract the PICMUS in-vivo RF/IQ dataset."""
    destination = Path(dest_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not force and picmus_in_vivo_available(destination):
        return resolve_picmus_in_vivo_root(destination)

    archive_path = destination / "in_vivo.zip"
    tmp_dir = Path(tempfile.mkdtemp(prefix="picmus-download-", dir=str(destination)))
    try:
        tmp_archive = tmp_dir / archive_path.name
        with urlopen(PICMUS_IN_VIVO_URL, timeout=timeout) as response, tmp_archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os_target = destination / "in_vivo"
        if force and os_target.exists():
            shutil.rmtree(os_target)
        shutil.move(str(tmp_archive), str(archive_path))
        with zipfile.ZipFile(archive_path) as zip_handle:
            zip_handle.extractall(destination)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return resolve_picmus_in_vivo_root(destination)


def default_picmus_case(root_dir: str | Path | None = None) -> str:
    """Return the preferred default case name for phase-retrieval demos."""
    available = list_picmus_rf_cases(root_dir)
    for case_name in _DEFAULT_CASES:
        if case_name in available:
            return case_name
    if available:
        return available[0]
    return _DEFAULT_CASES[1]
