"""Helpers for real transcranial hydrophone scans used in phase-retrieval demos."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import h5py
import numpy as np

ETH_TRANSCRANIAL_DATA_URL = (
    "https://www.research-collection.ethz.ch/bitstreams/"
    "f91ebcdd-af27-49ca-bf89-16eb85127553/download"
)
ETH_TRANSCRANIAL_SOURCE_URL = (
    "https://www.research-collection.ethz.ch/entities/dataset/"
    "77f541ab-2cc2-4274-a9f6-49c5475754ca"
)
_DEFAULT_CASES = (
    "Parietal_free_field_0_XY",
    "Frontal_free_field_0_XY",
    "Frontal_40_XY",
    "Parietal_0_XY",
)


@dataclass(frozen=True)
class TranscranialWaveformWindow:
    """One deterministic hydrophone trace window selected from a scan plane."""

    case_name: str
    row_index: int
    col_index: int
    start_index: int
    window_length: int
    trace: np.ndarray
    trace_energy: float
    dominant_frequency_bin: int
    scan_energy_map: np.ndarray


def _require_py7zr() -> Any:
    try:
        import py7zr
    except ImportError as exc:  # pragma: no cover - runtime only
        raise ImportError(
            "Transcranial phase-retrieval dataset support requires py7zr. "
            "Install the project dependencies before downloading the archive."
        ) from exc
    return py7zr


def resolve_transcranial_dataset_root(root_dir: str | Path | None = None) -> Path:
    """Resolve the extracted transcranial scan-data directory root."""
    base = Path(root_dir or Path("data") / "phase_retrieval").expanduser().resolve()
    candidates = (
        base,
        base / "transcranial",
        base / "eth_transcranial",
    )
    for candidate in candidates:
        if any((candidate / "Scan_data").glob("*.mat")):
            return candidate
    return base / "transcranial"


def list_transcranial_scan_cases(root_dir: str | Path | None = None) -> list[str]:
    """Return locally available transcranial scan-plane case names."""
    root = resolve_transcranial_dataset_root(root_dir)
    return sorted(path.stem for path in (root / "Scan_data").glob("*.mat"))


def transcranial_dataset_available(root_dir: str | Path | None = None) -> bool:
    """Return whether the transcranial hydrophone scan dataset is present locally."""
    return bool(list_transcranial_scan_cases(root_dir))


def default_transcranial_case(root_dir: str | Path | None = None) -> str:
    """Return the preferred case for the phase-retrieval demo."""
    available = list_transcranial_scan_cases(root_dir)
    for case_name in _DEFAULT_CASES:
        if case_name in available:
            return case_name
    if available:
        return available[0]
    return _DEFAULT_CASES[0]


def _case_path(root_dir: str | Path | None, case_name: str) -> Path:
    root = resolve_transcranial_dataset_root(root_dir)
    case_path = root / "Scan_data" / f"{case_name}.mat"
    if not case_path.exists():
        available = list_transcranial_scan_cases(root)
        raise FileNotFoundError(
            f"Transcranial scan case not found: {case_path}. Available cases: {available}"
        )
    return case_path


def load_transcranial_scan(root_dir: str | Path | None, *, case_name: str) -> np.ndarray:
    """Load one full hydrophone scan plane as a `(time, row, col)` array."""
    case_path = _case_path(root_dir, case_name)
    with h5py.File(case_path, "r") as h5_file:
        if "sigMat" not in h5_file:
            raise ValueError(f"MAT file {case_path} does not contain `sigMat`")
        sig = np.asarray(h5_file["sigMat"], dtype=np.float64)
    if sig.ndim != 3:
        raise ValueError(f"Expected a 3D `sigMat` array, found shape {sig.shape}")
    return sig


def _dominant_frequency_bin(sig: np.ndarray) -> int:
    spectrum = np.abs(np.fft.rfft(sig, axis=0))
    spatial_energy = spectrum.mean(axis=(1, 2))
    offset = 3 if spatial_energy.size > 4 else 1
    return int(np.argmax(spatial_energy[offset:]) + offset)


def load_transcranial_waveform_window(
    root_dir: str | Path | None,
    *,
    case_name: str,
    row_index: int,
    col_index: int,
    start_index: int,
    window_length: int,
) -> TranscranialWaveformWindow:
    """Load one explicitly indexed hydrophone waveform window."""
    if row_index < 0 or col_index < 0 or start_index < 0:
        raise ValueError("row_index, col_index, and start_index must be non-negative")
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    sig = load_transcranial_scan(root_dir, case_name=case_name)
    n_samples, n_rows, n_cols = sig.shape
    if row_index >= n_rows:
        raise ValueError(f"row_index must be < {n_rows}")
    if col_index >= n_cols:
        raise ValueError(f"col_index must be < {n_cols}")
    if start_index + window_length > n_samples:
        raise ValueError(
            f"Requested window exceeds trace length {n_samples}: start={start_index}, "
            f"length={window_length}"
        )

    trace = np.asarray(sig[:, row_index, col_index], dtype=np.float64)
    window = np.asarray(trace[start_index : start_index + window_length], dtype=np.float64)
    scan_energy = np.sum(sig**2, axis=0)
    return TranscranialWaveformWindow(
        case_name=case_name,
        row_index=int(row_index),
        col_index=int(col_index),
        start_index=int(start_index),
        window_length=int(window_length),
        trace=window,
        trace_energy=float(np.sum(window**2)),
        dominant_frequency_bin=_dominant_frequency_bin(sig),
        scan_energy_map=np.asarray(scan_energy, dtype=np.float64),
    )


def select_high_energy_hydrophone_window(
    root_dir: str | Path | None,
    *,
    case_name: str,
    window_length: int = 256,
    row_index: int | None = None,
    col_index: int | None = None,
) -> TranscranialWaveformWindow:
    """Select the strongest hydrophone trace and a pulse-centered waveform window."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    sig = load_transcranial_scan(root_dir, case_name=case_name)
    n_samples, n_rows, n_cols = sig.shape
    if window_length > n_samples:
        raise ValueError(f"window_length must be <= trace length {n_samples}, got {window_length}")

    scan_energy = np.sum(sig**2, axis=0)
    resolved_row: int
    resolved_col: int
    if row_index is None or col_index is None:
        indices = np.unravel_index(int(np.argmax(scan_energy)), scan_energy.shape)
        resolved_row = int(indices[0])
        resolved_col = int(indices[1])
    else:
        resolved_row = int(row_index)
        resolved_col = int(col_index)
        if resolved_row < 0 or resolved_row >= n_rows:
            raise ValueError(f"row_index must be between 0 and {n_rows - 1}")
        if resolved_col < 0 or resolved_col >= n_cols:
            raise ValueError(f"col_index must be between 0 and {n_cols - 1}")

    trace = np.asarray(sig[:, resolved_row, resolved_col], dtype=np.float64)
    peak_index = int(np.argmax(np.abs(trace)))
    left_margin = max(window_length // 3, 32)
    start_index = int(np.clip(peak_index - left_margin, 0, n_samples - window_length))

    return load_transcranial_waveform_window(
        root_dir,
        case_name=case_name,
        row_index=resolved_row,
        col_index=resolved_col,
        start_index=start_index,
        window_length=window_length,
    )


def download_transcranial_dataset(
    dest_dir: str | Path,
    *,
    force: bool = False,
    timeout: float = 180.0,
) -> Path:
    """Download and extract only the hydrophone scan MAT files needed for phase retrieval."""
    py7zr = _require_py7zr()
    destination = Path(dest_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not force and transcranial_dataset_available(destination):
        return resolve_transcranial_dataset_root(destination)

    archive_path = destination / "Transcranial_data.7z"
    extract_root = destination / "transcranial"
    tmp_dir = Path(tempfile.mkdtemp(prefix="transcranial-download-", dir=str(destination)))
    try:
        tmp_archive = tmp_dir / archive_path.name
        with urlopen(ETH_TRANSCRANIAL_DATA_URL, timeout=timeout) as response, tmp_archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if force and extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_archive), str(archive_path))
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            targets = [
                name
                for name in archive.getnames()
                if name.startswith("Scan_data/") and name.endswith(".mat")
            ]
            if not targets:
                raise ValueError("Transcranial archive does not contain any scan MAT files")
            archive.extract(path=extract_root, targets=targets)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return resolve_transcranial_dataset_root(destination)
