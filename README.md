# Ultrasound Imaging Toolkit

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
ultrasound-imaging-toolkit/
├── src/ultrasound/
│   ├── data/                 # Dataset loading and preprocessing
│   │   └── __init__.py       # BUSI dataset, synthetic data generation
│   ├── preprocessing/        # Image preprocessing modules
│   │   ├── speckle.py        # Speckle reduction filters
│   │   ├── enhancement.py    # Contrast enhancement methods
│   │   ├── denoising.py      # ADMM-TV, bilateral filtering
│   │   └── normalization.py  # Image normalization utilities
│   ├── models/               # Deep learning models
│   │   ├── unet.py           # U-Net, Attention U-Net
│   │   └── classifier.py     # CNN, ResNet classifiers
│   ├── utils/                # Utility functions
│   │   ├── io.py             # Image I/O, DICOM support
│   │   ├── metrics.py        # Dice, IoU, accuracy metrics
│   │   └── visualization.py  # Plotting and visualization
│   └── visualization/        # Visualization subpackage
├── notebooks/
│   └── ultrasound_demo.ipynb # Interactive demonstration
├── tests/                    # Unit tests
├── outputs/                  # Generated outputs
├── main.py                   # Main demo script
├── pyproject.toml           # Project configuration
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
jupyter notebook notebooks/ultrasound_demo.ipynb
```

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
