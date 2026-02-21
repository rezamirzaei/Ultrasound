# Ultrasound Imaging Toolkit

[![CI](https://github.com/rezamirzaei/ultrasound-imaging-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rezamirzaei/ultrasound-imaging-toolkit/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python toolkit for ultrasound image processing, analysis, and machine learning. Developed to demonstrate expertise in signal processing, medical imaging, and AI for the InPhase Solutions AS Senior Consultant position.

**Author:** Reza Mirzaeifard, PhD  
**Email:** reza.mirzaeifard@gmail.com  
**LinkedIn:** [linkedin.com/in/reza-mirzaeifard](https://www.linkedin.com/in/reza-mirzaeifard-0b28248b/)

---

## 🎯 Overview

This toolkit provides production-ready implementations of:

### Signal Processing
- **Speckle Reduction**: Lee filter, Frost filter, adaptive median filter
- **Contrast Enhancement**: CLAHE, histogram equalization, gamma correction
- **Denoising**: ADMM-based Total Variation, bilateral filtering, anisotropic diffusion

### Machine Learning
- **Segmentation**: U-Net, Attention U-Net for lesion segmentation
- **Classification**: Custom CNN, ResNet transfer learning for benign/malignant classification

### Optimization (PhD Expertise)
- **ADMM Implementation**: Alternating Direction Method of Multipliers for TV denoising
- **Proximal Operators**: Soft thresholding, projection methods
- **Convergence Analysis**: Primal and dual residual tracking

---

## 📁 Project Structure

```
inPhase/
├── src/ultrasound/           # Main package
│   ├── __init__.py
│   ├── demo.py               # CLI demo application
│   ├── data/                 # Dataset loading & synthetic generation
│   │   └── __init__.py       # BUSIDataset, create_sample_data
│   ├── preprocessing/        # Image preprocessing modules
│   │   ├── speckle.py        # Speckle reduction filters
│   │   ├── enhancement.py    # Contrast enhancement methods
│   │   ├── denoising.py      # ADMM-TV, bilateral filtering
│   │   └── normalization.py  # Image normalization utilities
│   ├── models/               # Deep learning models
│   │   ├── unet.py           # U-Net, Attention U-Net
│   │   └── classifier.py     # CNN, ResNet classifiers
│   ├── api/                  # REST API (FastAPI, MVC-style layering)
│   │   ├── controllers/      # HTTP controllers/routers
│   │   ├── services/         # Business logic services
│   │   ├── repositories/     # Data access abstractions
│   │   ├── database/         # SQLAlchemy ORM models/session
│   │   └── models/           # API schemas
│   └── utils/                # Utility functions
│       ├── io.py             # Image I/O, DICOM support
│       ├── metrics.py        # Dice, IoU, accuracy metrics
│       └── visualization.py  # Plotting and visualization
├── tests/                    # Unit tests (pytest)
├── scripts/                  # Data download utilities
├── notebooks/                # Jupyter demos & experiments
├── ui/                       # AngularJS MVC frontend
├── main.py                   # CLI entrypoint
├── pyproject.toml            # Project configuration
├── Makefile                  # Common dev commands
├── Dockerfile                # Container build
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rezamirzaei/ultrasound-imaging-toolkit.git
cd ultrasound-imaging-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Run Demo

```bash
python main.py
```

This will:
1. Generate synthetic ultrasound images
2. Demonstrate preprocessing techniques
3. Show ADMM optimization convergence
4. Display model architectures

### Jupyter Notebook

```bash
jupyter notebook notebooks/01_dataset_healthcheck.ipynb
```

Notebook suite and execution order are documented in `notebooks/README.md`.

### Run REST API + AngularJS UI

```bash
# Option 1
make api

# Option 2
python scripts/run_api.py

# Option 3 (Docker Compose)
docker compose up --build
```

Then open:
- UI: `http://localhost:8000/ui/index.html`
- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/api/v1/health`

Default sign-in accounts (change with `INPHASE_*_PASSWORD` env vars):
- `viewer / viewer123` (read-only dashboards + explorers)
- `analyst / analyst123` (includes preprocessing workflows)
- `admin / admin123` (includes operational error analytics)

Database:
- Default DB URL: `sqlite:///data/inphase.sqlite3`
- Override with `INPHASE_DATABASE_URL` (for Postgres/MySQL in production)

UI modules:
- Dashboard: project summary + quick navigation
- BUSI Explorer: browse images/masks by class and sample index
- BUSI Learning Monitor: run SQL-backed training and inspect train/test accuracy curves
- Preprocessing Lab: run Lee/Frost/CLAHE/ADMM-TV and compare metrics
- NDT Explorer: inspect sample metadata, fused defect detections, and sampled RF waveforms

Key API endpoints used by the UI:
- `GET /api/v1/dashboard/summary`
- `GET /api/v1/dashboard/readiness`
- `GET /api/v1/datasets/busi/samples/{class_name}/{sample_index}`
- `GET /api/v1/datasets/busi/training/latest?include_normal=false`
- `POST /api/v1/datasets/busi/training/run`
- `GET /api/v1/datasets/ndt/samples/{sample_name}/signal?max_points=1024`
- `POST /api/v1/preprocessing/preview`
- `POST /api/v1/ops/datasets/resync` (admin)
- `POST /api/v1/ops/datasets/resync` (admin)

Validation & reliability:
- BUSI and NDT datasets are persisted in SQLite (`data/inphase.sqlite3`) and served through SQLAlchemy ORM repositories.
- BUSI learning service trains from SQL-stored samples and persists run metrics + epoch curves.
- API error analytics are persisted in DB for durable operational dashboards.
- Repository outputs are normalized into typed Pydantic domain objects before entering services.
- API responses are strict Pydantic models (no NaN payload leaks to clients).
- NDT UI loads metadata and waveform independently, so waveform errors no longer break sample details.
- NDT defects are fused from metadata + waveform analysis (Hilbert envelope + adaptive peak detection).
- NDT signal endpoint exposes wall markers, total peaks, robust thickness method selection, confidence score, and CI95 uncertainty bounds.
- UI dependencies are vendored locally under `ui/vendor/` so sidebar routing works without external CDNs.
- API emits request IDs and captures operational error analytics under admin-only `/api/v1/ops/errors/*` endpoints.

Run tests in Docker explicitly:

```bash
docker compose --profile test run --rm test
```

Run browser E2E checks:

```bash
python -m playwright install chromium
make e2e
```

### Developer Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package in editable mode |
| `make dev` | Install with dev dependencies (pytest, black, mypy) |
| `make test` | Run test suite |
| `make lint` | Check code formatting |
| `make format` | Auto-format code |
| `make typecheck` | Run mypy type checking |
| `make demo` | Run CLI demo |
| `make api` | Start REST API + UI server |
| `make clean` | Remove build artifacts |
| `make docker-test` | Run tests inside Docker |
| `make e2e` | Run Playwright browser E2E tests |

---

## 📊 Dataset

### BUSI Dataset (Recommended)

The [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) contains:
- **780 images** in PNG format
- **3 classes**: Benign (437), Malignant (210), Normal (133)
- **Ground truth** segmentation masks

To use:
1. Download from Kaggle
2. Extract to `data/busi/`
3. Run the pipeline

### Synthetic Data

For testing without real data, the toolkit generates synthetic ultrasound-like images with:
- Speckle noise patterns
- Depth-dependent attenuation
- Simulated lesions (benign: oval, malignant: irregular)

---

## 🔧 Key Components

### 1. Speckle Reduction

```python
from ultrasound.preprocessing.speckle import SpeckleReducer

# Lee filter (MMSE estimator)
reducer = SpeckleReducer(method='lee', window_size=7)
denoised = reducer.reduce(ultrasound_image)

# Frost filter (exponential damping)
reducer = SpeckleReducer(method='frost', damping_factor=1.5)
denoised = reducer.reduce(ultrasound_image)
```

### 2. ADMM Total Variation Denoising

```python
from ultrasound.preprocessing.denoising import admm_tv_denoising

# Apply ADMM-based TV denoising
denoised, convergence = admm_tv_denoising(
    image,
    lambda_tv=0.1,  # Regularization weight
    rho=1.0,        # ADMM penalty parameter
    n_iter=50,
    verbose=True
)

# Analyze convergence
import matplotlib.pyplot as plt
plt.semilogy(convergence['primal_residuals'], label='Primal')
plt.semilogy(convergence['dual_residuals'], label='Dual')
plt.legend()
```

### 3. U-Net Segmentation

```python
from ultrasound.models.unet import UNet, dice_loss

# Create model
model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])

# Training loop
for images, masks, _ in dataloader:
    predictions = model(images)
    loss = dice_loss(predictions, masks)
    loss.backward()
    optimizer.step()
```

### 4. Classification

```python
from ultrasound.models.classifier import ResNetClassifier

# Transfer learning with ResNet-18
classifier = ResNetClassifier(
    num_classes=2,
    pretrained=True,
    freeze_backbone=True  # Only train classifier head
)

# Fine-tune later
classifier.unfreeze_backbone(num_layers=2)
```

---

## 📈 Technical Highlights

### ADMM Optimization Theory

The toolkit implements ADMM for Total Variation denoising, demonstrating expertise from my PhD research:

**Problem:**
$$\min_u \frac{1}{2}\|u - f\|_2^2 + \lambda \|Du\|_1$$

**ADMM Iterations:**
1. **u-update** (linear system): $(I + \rho D^T D)u = f + D^T(\rho z - y)$
2. **z-update** (soft thresholding): $z = S_{\lambda/\rho}(Du + y/\rho)$
3. **y-update** (dual ascent): $y = y + \rho(Du - z)$

This connects directly to my PhD work on non-convex optimization using ADMM.

### Speckle Noise Model

Ultrasound speckle follows a multiplicative model:
$$I(x,y) = R(x,y) \cdot n(x,y)$$

The Lee filter uses local statistics for MMSE estimation:
$$\hat{R} = \mu + W \cdot (I - \mu)$$
where $W = \text{Var}(R) / (\text{Var}(R) + \text{Var}(n))$

---

## 🏥 Applications

This toolkit is applicable to:

- **Breast Ultrasound**: Lesion detection and classification
- **Cardiac Ultrasound**: Chamber segmentation
- **Fetal Ultrasound**: Biometry measurements
- **Thyroid/Liver**: Nodule detection
- **Industrial NDT**: Defect detection in materials

---

## 📚 References

1. Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics. *IEEE TPAMI*.
2. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
3. Boyd et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*.
4. Al-Dhabyani et al. (2020). Dataset of breast ultrasound images. *Data in Brief*.

---

## 👨‍💻 About the Author

**Reza Mirzaeifard, PhD**

- PhD in Signal Processing/AI from NTNU, Norway
- Expertise in optimization algorithms (ADMM, proximal methods)
- Published 14+ papers in IEEE journals and conferences
- Experience with medical image processing and machine learning

This toolkit was developed to demonstrate ultrasound imaging expertise for the **InPhase Solutions AS** Senior Consultant position.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📬 Contact

- **Email**: reza.mirzaeifard@gmail.com
- **LinkedIn**: [Reza Mirzaeifard](https://www.linkedin.com/in/reza-mirzaeifard-0b28248b/)
- **Google Scholar**: [Publications](https://scholar.google.com/citations?user=NgVBhYsAAAAJ&hl=en)
