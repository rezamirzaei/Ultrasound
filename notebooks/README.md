# Notebooks

This folder now uses a split, reproducible notebook structure.

## Primary notebooks (recommended)

1. `01_dataset_healthcheck.ipynb`
   - Verifies local BUSI/NDT data availability.
   - Saves a health report to `outputs/notebooks/01_dataset_healthcheck/`.

2. `02_preprocessing_workbench.ipynb`
   - Runs speckle reduction, contrast enhancement, and ADMM-TV denoising.
   - Saves comparison figures and metrics report to `outputs/notebooks/02_preprocessing_workbench/`.

3. `03_models_and_metrics_smoke.ipynb`
   - Validates model forward passes and loss/metric APIs.
   - Saves confusion matrix and smoke report to `outputs/notebooks/03_models_and_metrics_smoke/`.

4. `04_mini_training_pipeline.ipynb`
   - Runs short deterministic training smoke loops on local BUSI data.
   - Saves curves and report to `outputs/notebooks/04_mini_training_pipeline/`.

5. `05_ndt_ascan_analysis.ipynb`
   - Performs A-scan envelope/spectrum/echo analysis on local NDT samples.
   - Saves figures and report to `outputs/notebooks/05_ndt_ascan_analysis/`.

6. `06_phase_retrieval_ultrasound.ipynb`
   - Demonstrates spectral init + Wirtinger Flow for amplitude-only reconstruction.
   - Saves convergence visuals and report to `outputs/notebooks/06_phase_retrieval_ultrasound/`.

7. `07_masked_proximal_decomposition.ipynb`
   - Demonstrates masked decomposition into smooth + sparse components for A-scan signals.
   - Saves decomposition outputs and report to `outputs/notebooks/07_masked_proximal_decomposition/`.

## Shared utilities

- `_notebook_utils.py`: project root resolution, deterministic seed setup, BUSI/NDT loaders, JSON reporting.

## Legacy notebooks

Previous large monolithic notebooks are preserved under `notebooks/legacy/` for reference.
They are not the recommended execution path.

### Legacy parity note

The modern suite now includes 7 primary notebooks (same count as legacy), but with cleaner scope
and deterministic execution.

## Regenerating `.ipynb` from `.py`

```bash
python -m jupytext --sync notebooks/*.py
```
