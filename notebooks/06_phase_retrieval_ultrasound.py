# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06. Phase Retrieval from Real Hydrophone Pulses
# Recover the missing phase of a measured ultrasound hydrophone waveform from STFT magnitude only.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
from _notebook_utils import (
    ensure_notebook_output_dir,
    ensure_src_on_path,
    save_json_report,
    set_reproducible_seed,
)

project_root = ensure_src_on_path()
from ultrasound.data.transcranial_phase_dataset import (  # noqa: E402
    default_transcranial_case,
    transcranial_dataset_available,
)
from ultrasound.workflows import run_phase_retrieval_transcranial  # noqa: E402

seed = set_reproducible_seed(42)
output_dir = ensure_notebook_output_dir("06_phase_retrieval_ultrasound")

print(f"Project root: {project_root}")
print(f"Seed: {seed}")
print(f"Output directory: {output_dir}")

# %%
phase_root = project_root / "data" / "phase_retrieval"
if not transcranial_dataset_available(phase_root):
    raise FileNotFoundError(
        "ETH transcranial hydrophone data is not available locally. "
        "Run `python scripts/download_transcranial_phase_retrieval.py` first."
    )

case_name = default_transcranial_case(phase_root)
result = run_phase_retrieval_transcranial(
    root_dir=str(phase_root),
    case_name=case_name,
    window_length=256,
    n_fft=80,
    hop_length=8,
    n_iter=120,
    seed=seed,
)
report = dict(result.report)
report["sample"] = f"ETH:{case_name}"
report["signal_metadata"] = result.signal_metadata
report

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

energy_map = result.scan_energy_map
metadata = result.signal_metadata or {}
row_index = int(metadata.get("row_index", 0))
col_index = int(metadata.get("col_index", 0))
axes[0, 0].imshow(energy_map, cmap="magma")
axes[0, 0].scatter([col_index], [row_index], s=80, c="cyan", marker="x")
axes[0, 0].set_title("Selected hydrophone location")
axes[0, 0].set_xlabel("Column")
axes[0, 0].set_ylabel("Row")

axes[0, 1].imshow(result.measured_spectrogram[::-1, :], aspect="auto", cmap="viridis")
axes[0, 1].set_title("Measured STFT magnitude")
axes[0, 1].set_xlabel("Frame")
axes[0, 1].set_ylabel("Frequency bin")

axes[0, 2].plot(result.residual_curve, color="#54A24B")
axes[0, 2].set_title("Consistency residual")
axes[0, 2].set_xlabel("Iteration")
axes[0, 2].set_ylabel("Relative error")
axes[0, 2].grid(alpha=0.25)

axes[1, 0].plot(result.true_signal, label="measured", color="#4C78A8")
axes[1, 0].plot(result.recovered_signal, label="recovered", color="#F58518", alpha=0.8)
axes[1, 0].set_title("Waveform")
axes[1, 0].grid(alpha=0.25)
axes[1, 0].legend()

axes[1, 1].plot(result.true_phase_spectrum, label="measured", color="#4C78A8")
axes[1, 1].plot(result.recovered_phase_spectrum, label="recovered", color="#F58518", alpha=0.8)
axes[1, 1].set_title("Phase spectrum")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend()

measured_norm = result.measured_spectrogram / (result.measured_spectrogram.max() + 1e-12)
reconstructed_norm = result.reconstructed_spectrogram / (result.reconstructed_spectrogram.max() + 1e-12)
axes[1, 2].scatter(measured_norm.ravel(), reconstructed_norm.ravel(), s=8, alpha=0.5, color="#B279A2")
axes[1, 2].plot([0, 1], [0, 1], "k--", lw=1.0)
axes[1, 2].set_title("Measured vs reconstructed magnitude")
axes[1, 2].set_xlabel("Measured")
axes[1, 2].set_ylabel("Reconstructed")
axes[1, 2].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "phase_retrieval_summary.png", dpi=140)
plt.close(fig)

# %%
save_json_report(output_dir / "phase_retrieval_report.json", report)
result.status
