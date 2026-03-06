# Ultrasound Imaging Toolkit

[![CI](https://github.com/rezamirzaei/ultrasound-imaging-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rezamirzaei/ultrasound-imaging-toolkit/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python toolkit for ultrasound imaging, NDT signal analysis, and ML-backed workflows. The repository includes a FastAPI backend, AngularJS UI, SQL-backed dataset services, notebook/workflow support, and deterministic test coverage across preprocessing, models, API, and background job execution.

Author: Reza Mirzaeifard, PhD  
Email: reza.mirzaeifard@gmail.com  
LinkedIn: [linkedin.com/in/reza-mirzaeifard](https://www.linkedin.com/in/reza-mirzaeifard-0b28248b/)

## Current Status

The repository is currently verified with:

- `ruff check .`
- `mypy ./`
- `pytest tests/ -q --cov=src/ultrasound --cov-fail-under=85`
- package build validation via `python -m build && python -m twine check dist/*`
- Docker test-image execution in CI

Current local verification snapshot:

- `344` passing tests
- `94.51%` total coverage
- CI gates on `ruff`, `mypy`, `pytest`, package build, Alembic migration upgrade, and Docker test execution

## What Is In The Repository

### Core processing

- speckle reduction: Lee, Frost, Wiener, adaptive median
- contrast enhancement: CLAHE, histogram equalization, gamma, logarithmic, adaptive enhancement
- denoising: TV denoising, ADMM-TV, bilateral filtering, anisotropic diffusion
- normalization, image I/O, metrics, and visualization helpers

### Models

- lesion segmentation with U-Net variants
- ultrasound classification with a custom CNN and ResNet transfer learning
- YOLO-oriented dataset prep and inference helpers for liver and BUSI workflows

### API and application layer

- FastAPI app with typed request/response schemas
- AngularJS UI served from the same app
- controller/service/repository split with protocol-based service boundaries
- SQLAlchemy persistence with Alembic migrations
- background job queue for training and dataset resync workloads
- Prometheus metrics and persisted error analytics
- DB-backed auth with PBKDF2 password hashes and revocable token sessions

### Data and workflows

- BUSI dataset loading and synthetic sample generation
- liver ultrasound detection dataset preparation and training flows
- NDT A-scan sample loading, wall echo analysis, thickness estimation, and defect fusion
- tested workflow modules under `src/ultrasound/workflows/`
- notebook wrappers under `notebooks/` that call library code instead of containing core logic directly

## Project Layout

```text
inPhase/
├── src/ultrasound/
│   ├── api/                  # FastAPI app, controllers, services, repositories, DB
│   ├── data/                 # BUSI + liver dataset loading/generation
│   ├── models/               # U-Net and classifier models
│   ├── preprocessing/        # Denoising, enhancement, normalization, speckle
│   ├── utils/                # I/O, metrics, visualization
│   ├── workflows/            # Reusable notebook-backed workflows
│   └── demo.py               # CLI demos
├── tests/                    # Pytest suite
├── notebooks/                # Thin notebook wrappers and notebook README
├── scripts/                  # API launcher and dataset utilities
├── ui/                       # AngularJS frontend
├── alembic/                  # Database migrations
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone https://github.com/rezamirzaei/ultrasound-imaging-toolkit.git
cd ultrasound-imaging-toolkit

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[notebooks]"
pip install -e ".[yolo]"
```

Notes:

- `.[dev]` installs `pytest`, `pytest-cov`, `ruff`, `mypy`, Playwright support, and related dev tools.
- `.[yolo]` installs Ultralytics for YOLO inference/training paths.
- the package declares `python >= 3.9`; CI runs on Python `3.11`

## Running The Project

### CLI demo

```bash
python main.py
```

This runs the demo entrypoint in [`main.py`](main.py) through [`src/ultrasound/demo.py`](src/ultrasound/demo.py).

### API and UI

Any of the following start the local API server:

```bash
make api
python scripts/run_api.py
python -m ultrasound.api.__main__
ultrasound-api
```

Then open:

- UI: `http://localhost:8000/ui/index.html`
- OpenAPI docs: `http://localhost:8000/docs`
- health: `http://localhost:8000/api/v1/health`
- metrics: `http://localhost:8000/metrics`

Default auth accounts:

- `viewer / viewer123`
- `analyst / analyst123`
- `admin / admin123`

These can be overridden with:

- `INPHASE_VIEWER_PASSWORD`
- `INPHASE_ANALYST_PASSWORD`
- `INPHASE_ADMIN_PASSWORD`
- `INPHASE_FORCE_DEFAULT_USERS=1` to re-seed default users from env values

Database defaults:

- default SQLite URL: `sqlite:///data/inphase.sqlite3`
- override with `INPHASE_DATABASE_URL`
- startup runs Alembic upgrades unless `INPHASE_SKIP_MIGRATIONS=1`
- legacy database auto-stamping can be disabled with `INPHASE_MIGRATION_AUTO_STAMP=0`

## Main UI / API Areas

The application currently exposes:

- dashboard summary and readiness
- BUSI dataset browsing and SQL-backed training metrics
- industrial dataset coverage, previews, and training flows
- NDT sample metadata, waveform preview, wall echo/thickness analysis, and defect exploration
- preprocessing preview lab
- YOLO liver detection lab
- YOLO BUSI ultrasound lab
- upload endpoints for BUSI and industrial samples
- operational error analytics and job queue endpoints

Representative endpoints:

- `GET /api/v1/dashboard/summary`
- `GET /api/v1/dashboard/readiness`
- `GET /api/v1/datasets/busi/samples/{class_name}/{sample_index}`
- `GET /api/v1/datasets/industrial/summary`
- `GET /api/v1/datasets/ndt/samples/{sample_name}`
- `GET /api/v1/datasets/ndt/samples/{sample_name}/signal`
- `POST /api/v1/preprocessing/preview`
- `POST /api/v1/datasets/busi/upload`
- `POST /api/v1/datasets/industrial/upload`
- `GET /api/v1/yolo/status`
- `GET /api/v1/yolo/liver/dataset/status`
- `POST /api/v1/yolo/liver/train`
- `GET /api/v1/yolo/ultrasound/busi/status`
- `POST /api/v1/yolo/ultrasound/busi/model/download`
- `GET /api/v1/learning/jobs`
- `POST /api/v1/learning/jobs/busi-training`
- `POST /api/v1/learning/jobs/datasets-resync`

## Datasets

### BUSI

The BUSI dataset is the primary medical-imaging dataset used here.

- breast ultrasound PNG images
- benign / malignant / normal classes
- lesion masks for segmentation-style and derived detection workflows

For local experiments:

1. download and extract BUSI into `data/busi/`, or
2. use the synthetic dataset generators in `ultrasound.data`

### Liver ultrasound detection

The liver detection flow expects the liver ultrasound detection dataset under `data/liver_ultrasound_detection/`. The code supports:

- dataset download and extraction
- annotation flattening to CSV
- flat-image preparation for YOLO training
- synthetic dataset fallback for smoke training

### NDT

NDT examples use `.npz` A-scan sample files under `data/ascan_signals/ndt_samples/`.

## Notebooks And Workflows

Notebooks are now thin wrappers around reusable library workflows.

Recommended notebook sequence:

1. `01_dataset_healthcheck.ipynb`
2. `02_preprocessing_workbench.ipynb`
3. `03_models_and_metrics_smoke.ipynb`
4. `04_mini_training_pipeline.ipynb`
5. `05_ndt_ascan_analysis.ipynb`
6. `06_phase_retrieval_ultrasound.ipynb`
7. `07_masked_proximal_decomposition.ipynb`

See [notebooks/README.md](notebooks/README.md) for details and output locations.

## Development Commands

Available `Makefile` targets:

| Command | Purpose |
|---|---|
| `make install` | install package in editable mode |
| `make dev` | install dev dependencies |
| `make test` | run tests |
| `make lint` | run `black --check` and `isort --check-only` |
| `make format` | format with Black and isort |
| `make typecheck` | run mypy |
| `make demo` | run CLI demo |
| `make api` | start API/UI locally |
| `make db-upgrade` | apply Alembic migrations |
| `make db-downgrade` | roll back one migration |
| `make db-revision m="msg"` | create autogenerated Alembic revision |
| `make docker-test` | run tests in Docker Compose |
| `make e2e` | run Playwright browser tests |
| `make clean` | remove caches and build artifacts |

Commands used by CI:

```bash
python -m ruff check .
python -m mypy ./
pytest tests/ -q --cov=src/ultrasound --cov-report=term-missing --cov-fail-under=85
INPHASE_DATABASE_URL=sqlite:////tmp/inphase-ci.sqlite3 alembic upgrade head
python -m build && python -m twine check dist/*
docker build --target test -t ultrasound-toolkit:test .
docker run --rm ultrasound-toolkit:test
```

## CI/CD

GitHub Actions currently runs:

- change detection to avoid unnecessary jobs
- backend quality checks
- Alembic migration verification
- package build validation
- Docker test image build and execution
- frontend smoke checks when UI files change

Workflow file: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## License

MIT
