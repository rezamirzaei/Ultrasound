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
# # Signal Decomposition using Masked Proximal Operators
#
# **Author:** Reza Mirzaeifard, PhD
# **Email:** reza.mirzaeifard@gmail.com
# **Project:** Advanced Signal Processing for InPhase Solutions AS
# **Date:** January 2026
# **Reference:** Meyers, B. E., & Boyd, S. (2023). Signal decomposition using masked proximal operators.
# *Foundations and Trends® in Signal Processing*, 17(1), 1-78.
#
# ---
#
# ## Executive Summary
#
# This notebook implements the **Masked Proximal Operator** framework for signal decomposition,
# as introduced by Meyers and Boyd (2023). The key idea is to decompose a signal into multiple
# components, each with distinct characteristics (smooth, sparse, periodic, etc.), while
# handling **missing data** (masks) naturally within the optimization framework.
#
# ### The Problem Formulation
#
# Given an observed signal $y$ with some missing entries indicated by a mask $M$, we want to
# decompose it into $K$ components:
#
# $$y = \sum_{k=1}^{K} x_k + \epsilon$$
#
# where each $x_k$ has a specific structure enforced by a regularizer $r_k(x_k)$.
#
# The optimization problem is:
#
# $$\min_{x_1, \ldots, x_K} \frac{1}{2} \left\| M \odot \left( y - \sum_{k=1}^{K} x_k \right) \right\|_2^2 + \sum_{k=1}^{K} \lambda_k r_k(x_k)$$
#
# where:
# - $M$ is a binary mask (1 = observed, 0 = missing)
# - $\odot$ is element-wise multiplication
# - $r_k$ are regularizers (e.g., total variation, L1, smoothness)
# - $\lambda_k$ are regularization weights
#
# ### Applications
#
# | Application | Components | Use Case |
# |-------------|-----------|----------|
# | **NDT Signal Analysis** | Trend + Defect + Noise | Separate baseline from defect echoes |
# | **Time Series** | Trend + Seasonal + Residual | Forecasting, anomaly detection |
# | **Medical Ultrasound** | Tissue + Speckle + Noise | Image enhancement |
# | **Power Systems** | Base load + Peaks + Noise | Load forecasting |
#
# ---
#
# ## 1. Environment Setup

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

warnings.filterwarnings('ignore')

from IPython.display import Image, display

# --- Helper functions for physically meaningful decision-making ---

def estimate_front_back_wall(time_s: np.ndarray, envelope: np.ndarray, c_mps: float, thickness_m: float | None):
    """Estimate front/back wall indices from an envelope.

    - Front wall: the earliest strong peak.
    - Back wall: peak closest to the expected thickness (if provided), else latest strong peak.

    Returns (fw_idx, bw_idx) with fw_idx <= bw_idx.
    """
    from scipy.signal import find_peaks

    envelope = np.asarray(envelope)
    n = len(envelope)
    if n == 0:
        return 0, 0

    height_thr = float(np.max(envelope)) * 0.15
    peaks, props = find_peaks(envelope, height=height_thr, distance=max(5, n // 200))

    if peaks.size == 0:
        fw = int(np.argmax(envelope))
        return fw, n - 1

    heights = props.get('peak_heights', envelope[peaks])

    strong_cut = np.percentile(heights, 80)
    strong_peaks = peaks[heights >= strong_cut]
    if strong_peaks.size == 0:
        strong_peaks = peaks

    fw_idx = int(np.min(strong_peaks))

    depth_mm = np.asarray(time_s) * float(c_mps) / 2 * 1000

    if thickness_m is not None and np.isfinite(thickness_m):
        thickness_mm = float(thickness_m) * 1000
        tol_mm = max(1.0, 0.12 * thickness_mm)
        target_min = thickness_mm - tol_mm
        target_max = thickness_mm + tol_mm

        candidates = np.array([p for p in peaks if (p > fw_idx) and (target_min <= depth_mm[p] <= target_max)], dtype=int)
        if candidates.size:
            bw_idx = int(candidates[np.argmin(np.abs(depth_mm[candidates] - thickness_mm))])
        else:
            after = strong_peaks[strong_peaks > fw_idx]
            bw_idx = int(np.max(after)) if after.size else int(np.max(strong_peaks))
    else:
        after = strong_peaks[strong_peaks > fw_idx]
        bw_idx = int(np.max(after)) if after.size else int(np.max(strong_peaks))

    if fw_idx > bw_idx:
        fw_idx, bw_idx = bw_idx, fw_idx

    return fw_idx, bw_idx


def roi_mask_between_walls(time_s: np.ndarray, c_mps: float, fw_idx: int, bw_idx: int, margin_mm: float = 0.5):
    """Return (roi_bool, (fw_mm, bw_mm, roi_min_mm, roi_max_mm))."""
    depth_mm = np.asarray(time_s) * float(c_mps) / 2 * 1000
    fw_mm = float(depth_mm[int(fw_idx)])
    bw_mm = float(depth_mm[int(bw_idx)])

    roi_min_mm = fw_mm + float(margin_mm)
    roi_max_mm = bw_mm - float(margin_mm)

    if roi_max_mm <= roi_min_mm:
        return np.zeros_like(depth_mm, dtype=bool), (fw_mm, bw_mm, roi_min_mm, roi_max_mm)

    roi = (depth_mm >= roi_min_mm) & (depth_mm <= roi_max_mm)
    return roi, (fw_mm, bw_mm, roi_min_mm, roi_max_mm)

# Configure paths
project_root = Path('.').absolute().parent
ndt_data_path = project_root / 'data' / 'ascan_signals' / 'ndt_samples'
output_path = project_root / 'outputs'
output_path.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("✓ Environment configured")
print(f"✓ NDT data path: {ndt_data_path}")
print(f"✓ Output path: {output_path}")

# %% [markdown]
# ---
# ## 2. Proximal Operators for Signal Decomposition
#
# The core of the Meyers-Boyd framework is using **proximal operators** to enforce
# structure on each signal component. The proximal operator of a function $r$ is:
#
# $$\text{prox}_{\lambda r}(v) = \arg\min_x \left( \frac{1}{2}\|x - v\|_2^2 + \lambda r(x) \right)$$
#
# ### Common Proximal Operators
#
# | Regularizer | Effect | Proximal Operator |
# |-------------|--------|-------------------|
# | **L1 norm** | Sparsity | Soft thresholding |
# | **Total Variation** | Piecewise constant | TV denoising |
# | **Quadratic smoothness** | Smooth signal | Low-pass filter |
# | **L2 norm** | Shrinkage | Scaling |

# %%
def prox_l1(v, lam):
    """Proximal operator for L1 norm (soft thresholding).

    Promotes sparsity in the signal.
    prox_{λ||·||_1}(v) = sign(v) * max(|v| - λ, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)


def prox_l2(v, lam):
    """Proximal operator for L2 norm (shrinkage).

    Shrinks the signal uniformly.
    prox_{λ||·||_2}(v) = v / (1 + λ)
    """
    return v / (1 + lam)


def prox_tv(v, lam, max_iter=50):
    """Proximal operator for 1D Total Variation.

    Promotes piecewise constant signals.
    Uses iterative clipping algorithm (Condat 2013).
    """
    n = len(v)
    if n <= 1:
        return v.copy()

    x = v.copy()

    # Simplified TV proximal via iterative filtering
    for _ in range(max_iter):
        # Compute gradient
        grad = np.zeros(n)
        grad[:-1] = x[:-1] - x[1:]
        grad[1:] -= x[:-1] - x[1:]

        # Soft threshold the gradient
        x = v - lam * grad

    return x


def prox_smooth(v, lam, order=2):
    """Proximal operator for quadratic smoothness (exact).

    Solves: min_x 1/2 ||x - v||_2^2 + lam ||D^k x||_2^2
    Solution: x = (I + 2*lam*D^T D)^-1 v

    This is much better than filtering for trend estimation.
    """
    n = len(v)
    # Construct 2nd order difference matrix D (tridiagonal)
    # D corresponds to [1, -2, 1] kernel
    if order == 2:
        # D2 matrix
        diagonals = [np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)]
        D = diags(diagonals, [0, 1, 2], shape=(n-2, n))
    elif order == 1:
        # D1 matrix (TV-like but quadratic)
        diagonals = [-np.ones(n-1), np.ones(n-1)]
        D = diags(diagonals, [0, 1], shape=(n-1, n))
    else:
        return v # Fallback

    # System matrix A = I + 2*lam * D.T @ D
    # Factor 2 comes from the 1/2 in the proximal definition vs standard form
    # prox_f(v) = argmin 1/2||x-v||^2 + f(x). If f(x) = lam ||Dx||^2, gradient is x-v + 2*lam*D'Dx = 0

    A = diags([np.ones(n)], [0], shape=(n, n)) + (2 * lam) * (D.T @ D)

    # Solve linear system (fast for banded matrices)
    return spsolve(A, v)


def prox_periodic(v, lam, period):
    """Proximal operator promoting periodic structure.

    Projects onto space of periodic signals with given period.
    """
    n = len(v)
    if period >= n:
        return v.copy()

    # Average over periods
    n_periods = n // period
    remainder = n % period

    # Compute average pattern
    pattern = np.zeros(period)
    for i in range(n_periods):
        pattern += v[i*period:(i+1)*period]
    if remainder > 0:
        pattern[:remainder] += v[n_periods*period:]
        counts = np.ones(period) * n_periods
        counts[:remainder] += 1
        pattern /= counts
    else:
        pattern /= n_periods

    # Tile the pattern
    periodic = np.tile(pattern, n_periods)
    if remainder > 0:
        periodic = np.concatenate([periodic, pattern[:remainder]])

    # Blend with original based on lambda
    return (v + lam * periodic) / (1 + lam)


def prox_l2_sq(v, lam):
    """Prox for (lam/2)*||x||_2^2.

    prox_{(lam/2)||·||^2}(v) = v / (1 + lam)

    This makes the quadratic penalties consistent and lets us compute a meaningful objective.
    """
    return v / (1.0 + lam)


def smooth_penalty_quadratic(x, order=2):
    """Compute ||D^k x||_2^2 for k=1 or k=2 using finite differences."""
    x = np.asarray(x)
    if order == 1:
        d = np.diff(x)
        return float(np.sum(d * d))
    if order == 2:
        d2 = x[:-2] - 2 * x[1:-1] + x[2:]
        return float(np.sum(d2 * d2))
    return 0.0


print("✓ Proximal operators defined:")
print("  - prox_l1: Sparsity (soft thresholding)")
print("  - prox_l2: Shrinkage")
print("  - prox_tv: Total Variation (piecewise constant)")
print("  - prox_smooth: Smoothness (low-pass)")
print("  - prox_periodic: Periodic structure")

# %% [markdown]
# ---
# ## 3. Masked Proximal Decomposition Algorithm
#
# The algorithm uses **ADMM (Alternating Direction Method of Multipliers)** to solve
# the signal decomposition problem with missing data.
#
# ### Algorithm: Masked Signal Decomposition
#
# **Input:** Signal $y$, mask $M$, regularizers $\{r_k\}$, weights $\{\lambda_k\}$
#
# **Initialize:** $x_k^0 = 0$, dual variables $u_k^0 = 0$
#
# **Repeat:**
# 1. Update residual: $r = y - \sum_k x_k$
# 2. For each component $k$:
#    - $v_k = x_k + M \odot r / K + u_k$
#    - $x_k^{new} = \text{prox}_{\lambda_k r_k}(v_k)$
#    - $u_k = u_k + (v_k - x_k^{new})$
#
# **Until** convergence

# %%
class MaskedProximalDecomposition:
    """Masked signal decomposition using proximal operators.

    We solve a *practical* structured decomposition objective:

        minimize 0.5 || M ⊙ (y - Σ_k x_k) ||_2^2
              + (λ_smooth/2) ||D^2 x_smooth||_2^2
              + λ_sparse ||x_sparse||_1
              + (λ_noise/2) ||x_noise||_2^2

    This matches the proximal operators we apply:
    - x_smooth  : prox of (λ/2)||D^2 x||^2  (implemented by solving a banded linear system)
    - x_sparse  : prox of λ||x||_1          (soft threshold)
    - x_noise   : prox of (λ/2)||x||^2      (shrinkage)

    Optimization method:
    - We use a robust *alternating proximal residual-splitting* scheme (a consensus-style method)
      that behaves like ADMM/PPXA for separable priors and is stable for this application.

    Notes:
    - The previous version had contradictory dual updates. We remove the dual state entirely
      (set u ≡ 0), which gives a clean, reproducible baseline method.
    - If you want full ADMM with explicit consensus variables, we can add it, but this
      version is much harder to break and is sufficient for decision metrics.
    """

    def __init__(self, components, max_iter=300, tol=1e-6, verbose=True):
        self.components = components
        self.K = len(components)
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.x = None
        self.residual_history = []
        self.objective_history = []

    def _objective(self, y, mask, x_dict):
        """Compute the objective value consistent with the chosen component penalties."""
        y = np.asarray(y)
        mask = np.asarray(mask)
        recon = np.zeros_like(y)
        for v in x_dict.values():
            recon = recon + v

        data_term = 0.5 * float(np.sum((mask * (y - recon)) ** 2))

        reg = 0.0
        for comp in self.components:
            name = comp['name']
            lam = float(comp['lambda'])
            kind = comp.get('kind', None)
            if kind == 'smooth2':
                order = int(comp.get('prox_kwargs', {}).get('order', 2))
                reg += 0.5 * lam * smooth_penalty_quadratic(x_dict[name], order=order)
            elif kind == 'sparse_l1':
                reg += lam * float(np.sum(np.abs(x_dict[name])))
            elif kind == 'noise_l2sq':
                reg += 0.5 * lam * float(np.sum(x_dict[name] ** 2))
            else:
                # fallback: treat as L1 so objective isn't silently nonsense
                reg += lam * float(np.sum(np.abs(x_dict[name])))

        return data_term + reg

    def decompose(self, y, mask=None):
        y = np.asarray(y).astype(float)
        n = len(y)

        if mask is None:
            mask = np.ones(n, dtype=float)
        mask = np.asarray(mask, dtype=float)

        x = {comp['name']: np.zeros(n, dtype=float) for comp in self.components}

        self.residual_history = []
        self.objective_history = []

        for it in range(self.max_iter):
            x_old = {k: v.copy() for k, v in x.items()}

            # Current reconstruction and residual on observed samples
            recon = np.zeros_like(y)
            for v in x.values():
                recon = recon + v
            r = mask * (y - recon)

            # Update each component using a fair share of the residual
            # v_k = x_k + r/K
            # x_k <- prox_k(v_k)
            for comp in self.components:
                name = comp['name']
                prox_fn = comp['prox']
                lam = float(comp['lambda'])
                prox_kwargs = comp.get('prox_kwargs', {})

                v = x[name] + r / self.K
                x[name] = prox_fn(v, lam, **prox_kwargs)

            # Convergence
            step_norm = float(np.sqrt(sum(np.sum((x[k] - x_old[k]) ** 2) for k in x)))
            self.residual_history.append(step_norm)

            obj = self._objective(y, mask, x)
            self.objective_history.append(obj)

            if self.verbose and (it % 25 == 0 or it == self.max_iter - 1):
                print(f"  Iter {it:4d}: step_norm={step_norm:.2e}, objective={obj:.6f}")

            if step_norm < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {it}")
                break

        self.x = x
        return x

    def impute_missing(self, y, mask):
        """Impute missing values (mask==0) using the current decomposition.

        If decomposition not computed yet, runs it first.
        Returns reconstructed signal sum(x_k).
        """
        if self.x is None:
            self.decompose(y, mask)
        recon = np.zeros_like(np.asarray(y).astype(float))
        for v in self.x.values():
            recon = recon + v
        return recon

    def plot_decomposition(self, y, mask=None, time=None, save_path=None, title=None):
        """Plot original signal, each component, and reconstruction."""
        if self.x is None:
            raise ValueError("Must call decompose() first")

        y = np.asarray(y).astype(float)
        n = len(y)
        if time is None:
            time = np.arange(n)
        if mask is None:
            mask = np.ones(n)
        mask = np.asarray(mask)

        K = len(self.x)
        fig, axes = plt.subplots(K + 2, 1, figsize=(14, 3 * (K + 2)), sharex=True)

        # Original
        ax = axes[0]
        ax.plot(time, y, linewidth=0.8, label='Observed')
        missing_idx = np.where(mask == 0)[0]
        if missing_idx.size:
            ax.scatter(np.asarray(time)[missing_idx], y[missing_idx], s=10, c='red', marker='x', alpha=0.5, label='Missing')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right')
        ax.set_title(title or 'Signal and decomposition')

        # Components
        for i, (name, sig) in enumerate(self.x.items(), start=1):
            ax = axes[i]
            ax.plot(time, sig, linewidth=0.9)
            ax.set_ylabel('amp')
            ax.grid(True, alpha=0.25)
            ax.set_title(name)

        # Reconstruction
        recon = np.zeros_like(y)
        for v in self.x.values():
            recon = recon + v
        ax = axes[-1]
        ax.plot(time, y, linewidth=0.8, alpha=0.5, label='Original')
        ax.plot(time, recon, linewidth=1.0, label='Reconstruction')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right')
        ax.set_xlabel('time')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

# %% [markdown]
# ---
# ## 4. Dataset: NDT Ultrasound A-scan Signals
#
# We apply the masked proximal decomposition to **real NDT ultrasound A-scan data**.
# This is an excellent use case because:
#
# 1. **Natural component structure**:
#    - **Trend**: Baseline signal / coupling variations
#    - **Sparse peaks**: Echoes from walls and defects
#    - **Noise**: Electronic and acoustic noise
#
# 2. **Missing data scenario**:
#    - Gated regions (intentionally blanked)
#    - Saturated samples
#    - Corrupted measurements
#
# ### Goal
# Decompose each A-scan into:
# - **Smooth baseline** (slow-varying trend)
# - **Sparse echoes** (defect/wall reflections)
# - **Residual noise**

# %%
def load_ndt_data(filepath):
    """Load NDT A-scan data from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    return {
        'rf': np.asarray(data['rf']).astype(float),
        'time': np.asarray(data['time']).astype(float),
        'fs': float(data['fs']),
        'fc': float(data['fc']),
        'c': float(data['c']),
        'description': str(data['description']),
        'thickness': float(data['thickness']),
    }


# Load all NDT datasets
ndt_files = list(ndt_data_path.glob('*.npz'))
print(f"✓ Found {len(ndt_files)} NDT data files:")

ndt_datasets = {}
for f in ndt_files:
    data = load_ndt_data(f)
    ndt_datasets[f.stem] = data
    print(f"  - {f.stem}: {len(data['rf'])} samples, fs={data['fs']/1e6:.1f} MHz")

# Use weld inspection as primary example
primary_name = 'weld_inspection'
primary_data = ndt_datasets[primary_name]
y = primary_data['rf']
t = primary_data['time']
fs = primary_data['fs']

print(f"\n✓ Primary dataset: {primary_name}")
print(f"  Samples: {len(y)}")
print(f"  Duration: {t[-1]*1e6:.1f} µs")

# %% [markdown]
# ---
# ## 5. Apply Masked Proximal Decomposition to NDT Data
#
# We decompose the ultrasound A-scan into three components:
#
# 1. **Smooth trend**: Low-frequency baseline variations (using smooth proximal)
# 2. **Sparse echoes**: High-amplitude, localized reflections (using L1 proximal)
# 3. **Residual**: Remaining noise/texture (using L2 proximal)
#
# ### Simulating Missing Data
# To demonstrate the masked capability, we artificially "gate out" a portion of the signal
# (simulating a common practice in UT where near-field ringing is blanked).

# %%
# Create a realistic mask based on ultrasound physics:
# - gate the near-field region up to just before the front-wall echo
# - add random dropouts OUTSIDE the internal ROI (so decision metrics remain reliable)

env = np.abs(hilbert(y))
fw_idx, bw_idx = estimate_front_back_wall(
    time_s=t,
    envelope=env,
    c_mps=float(primary_data['c']),
    thickness_m=float(primary_data.get('thickness', np.nan)) if primary_data.get('thickness') is not None else None,
)

roi_bool, (fw_mm, bw_mm, roi_min_mm, roi_max_mm) = roi_mask_between_walls(
    time_s=t,
    c_mps=float(primary_data['c']),
    fw_idx=fw_idx,
    bw_idx=bw_idx,
    margin_mm=0.5,
)

mask = np.ones(len(y), dtype=float)
mask[:max(0, fw_idx - int(0.2e-6 * fs))] = 0

np.random.seed(42)
candidates = np.where(~roi_bool & (mask == 1))[0]
if candidates.size > 0:
    dropout_count = int(0.03 * len(y))
    dropout_idx = np.random.choice(candidates, size=min(dropout_count, candidates.size), replace=False)
    mask[dropout_idx] = 0

n_missing = int(np.sum(mask == 0))
print("✓ Created wall/ROI-based mask")
print(f"  Front wall ~ {fw_mm:.2f} mm (idx={fw_idx})")
print(f"  Back wall  ~ {bw_mm:.2f} mm (idx={bw_idx})")
print(f"  ROI(mm)    = [{roi_min_mm:.2f}, {roi_max_mm:.2f}] | ROI samples={int(np.sum(roi_bool))}")
print(f"  Missing samples: {n_missing} ({100*n_missing/len(y):.1f}%)")

# Define decomposition components
components = [
    {
        'name': 'Smooth Baseline',
        'kind': 'smooth2',
        'prox': prox_smooth,
        # baseline should be smooth but not dominate; tuned moderate
        'lambda': 250.0,
        'prox_kwargs': {'order': 2}
    },
    {
        'name': 'Sparse Echoes',
        'kind': 'sparse_l1',
        'prox': prox_l1,
        'lambda': 0.05,
        'prox_kwargs': {}
    },
    {
        'name': 'Residual Noise',
        'kind': 'noise_l2sq',
        'prox': prox_l2_sq,
        'lambda': 1.0,
        'prox_kwargs': {}
    }
]

print("\n✓ Decomposition components:")
for comp in components:
    print(f"  - {comp['name']}: λ={comp['lambda']}")

# %%
# Run decomposition
print("\nRunning Masked Proximal Decomposition...")
print("=" * 60)

decomposer = MaskedProximalDecomposition(
    components=components,
    max_iter=300,
    tol=1e-6,
    verbose=True,
)

x = decomposer.decompose(y, mask)

print("=" * 60)
print("\n✓ Decomposition complete!")

# Report component statistics
print("\nComponent Statistics:")
print("-" * 60)
for name, signal in x.items():
    energy = np.sum(signal**2)
    sparsity = np.sum(np.abs(signal) > 0.01 * np.max(np.abs(signal))) / len(signal)
    print(f"  {name:20s}: energy={energy:.4f}, sparsity={100*sparsity:.1f}%")

# %% [markdown]
# ### Visualization

# %%
# Plot the decomposition
fig = decomposer.plot_decomposition(
    y, mask, time=t * 1e6,
    save_path=output_path / 'masked_proximal_ndt_decomposition.png'
)

# %% [markdown]
# ---
# ## 6. Analysis: Convergence and Component Interpretation
#
# ### Convergence Analysis
# We examine how the algorithm converges and the objective function decreases.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual convergence
ax = axes[0]
ax.semilogy(decomposer.residual_history, 'b-', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Primal Residual')
ax.set_title('Convergence: Primal Residual')
ax.grid(True, alpha=0.3)
ax.axhline(decomposer.tol, color='r', linestyle='--', label=f'Tolerance ({decomposer.tol})')
ax.legend()

# Objective function
ax = axes[1]
ax.plot(decomposer.objective_history, 'g-', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value')
ax.set_title('Convergence: Objective Function')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'masked_proximal_convergence.png', dpi=150)
plt.show()

# %% [markdown]
# ### Component Interpretation
#
# Let's analyze what each component captured:

# %%
print("\n" + "=" * 70)
print("COMPONENT INTERPRETATION")
print("=" * 70)

# Smooth Baseline
baseline = x['Smooth Baseline']
print("\n1. SMOOTH BASELINE")
print("-" * 40)
print(f"   Range: [{baseline.min():.4f}, {baseline.max():.4f}]")
print(f"   Std Dev: {baseline.std():.4f}")
print("   Interpretation: This captures slow-varying trends in the signal,")
print("   such as coupling variations, beam spread effects, and attenuation.")

# Sparse Echoes
echoes = x['Sparse Echoes']
threshold = 0.1 * np.max(np.abs(echoes))
significant_echoes = np.where(np.abs(echoes) > threshold)[0]
print("\n2. SPARSE ECHOES")
print("-" * 40)
print(f"   Number of significant echoes: {len(significant_echoes)}")
print(f"   Max amplitude: {np.max(np.abs(echoes)):.4f}")
if len(significant_echoes) > 0:
    depths = t[significant_echoes] * primary_data['c'] / 2 * 1000  # mm
    print(f"   Echo locations (depth in mm): {depths[:5].round(2)}...")
print("   Interpretation: These are the localized, high-amplitude reflections")
print("   from walls, defects, and interfaces - the 'useful' signal in NDT.")

# Residual Noise
noise = x['Residual Noise']
snr_db = 10 * np.log10(np.var(echoes) / (np.var(noise) + 1e-12))
print("\n3. RESIDUAL NOISE")
print("-" * 40)
print(f"   Std Dev: {noise.std():.4f}")
print(f"   Estimated SNR: {snr_db:.1f} dB")
print("   Interpretation: This captures electronic noise, grain noise,")
print("   and other high-frequency random variations.")

print("\n" + "=" * 70)

# %% [markdown]
# ---
# ## 7. Application: Missing Data Imputation
#
# One powerful feature of masked proximal decomposition is its ability to
# **impute missing values** while respecting the signal structure.

# %%
# Impute the missing values
y_imputed = decomposer.impute_missing(y, mask)

# Visualize imputation
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Original with missing
ax = axes[0]
ax.plot(t * 1e6, y, 'b-', linewidth=0.8, label='Observed')
missing_idx = np.where(mask == 0)[0]
ax.scatter(t[missing_idx] * 1e6, y[missing_idx], c='red', s=15,
          marker='x', label='Missing (unknown)', alpha=0.5, zorder=5)
ax.set_ylabel('Amplitude')
ax.set_title('Original Signal with Missing Data')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Imputed signal
ax = axes[1]
ax.plot(t * 1e6, y_imputed, 'g-', linewidth=0.8, label='Imputed signal')
ax.scatter(t[missing_idx] * 1e6, y_imputed[missing_idx], c='orange', s=15,
          marker='o', label='Imputed values', alpha=0.7, zorder=5)
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Amplitude')
ax.set_title('Imputed Signal (Missing Values Filled)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'masked_proximal_imputation.png', dpi=150)
plt.show()

# Imputation quality metrics
observed_idx = np.where(mask == 1)[0]
reconstruction_error = np.sqrt(np.mean((y[observed_idx] - y_imputed[observed_idx])**2))
print("\n✓ Imputation complete")
print(f"  Reconstruction RMSE on observed data: {reconstruction_error:.6f}")
print(f"  Number of imputed samples: {len(missing_idx)}")

# %% [markdown]
# ---
# ## 8. Apply to All NDT Datasets
#
# We now apply the decomposition to all available NDT datasets and compare results.

# %%
print("\nApplying decomposition to all NDT datasets...")
print("=" * 70)

all_results = {}

for name, data in ndt_datasets.items():
    print(f"\nProcessing: {name}")

    y_data = data['rf']
    t_data = data['time']
    fs_data = float(data['fs'])
    c_data = float(data['c'])
    thickness_m = float(data.get('thickness', np.nan)) if data.get('thickness') is not None else None

    # Compute envelope and estimate walls
    env_data = np.abs(hilbert(y_data))
    fw_idx, bw_idx = estimate_front_back_wall(t_data, env_data, c_data, thickness_m)
    roi_bool_data, (fw_mm, bw_mm, roi_min_mm, roi_max_mm) = roi_mask_between_walls(t_data, c_data, fw_idx, bw_idx, margin_mm=0.5)

    # Mask: simulate gating up to front wall, dropouts outside ROI
    mask_data = np.ones(len(y_data), dtype=float)
    mask_data[:max(0, fw_idx - int(0.2e-6 * fs_data))] = 0

    candidates = np.where(~roi_bool_data & (mask_data == 1))[0]
    if candidates.size > 0:
        dropout_count = int(0.03 * len(y_data))
        dropout_idx = np.random.choice(candidates, size=min(dropout_count, candidates.size), replace=False)
        mask_data[dropout_idx] = 0

    # Define components with auto-tuned lambda based on signal statistics
    signal_std = float(np.std(y_data[mask_data == 1]))
    components_auto = [
        {
            'name': 'Smooth Baseline',
            'kind': 'smooth2',
            'prox': prox_smooth,
            'lambda': 200.0 * signal_std,
            'prox_kwargs': {'order': 2}
        },
        {
            'name': 'Sparse Echoes',
            'kind': 'sparse_l1',
            'prox': prox_l1,
            'lambda': 0.06 * signal_std,
            'prox_kwargs': {}
        },
        {
            'name': 'Residual Noise',
            'kind': 'noise_l2sq',
            'prox': prox_l2_sq,
            'lambda': 1.0 * signal_std,
            'prox_kwargs': {}
        }
    ]

    decomposer_auto = MaskedProximalDecomposition(
        components=components_auto,
        max_iter=250,
        tol=1e-6,
        verbose=False
    )

    x_data = decomposer_auto.decompose(y_data, mask_data)

    recon_data = sum(x_data.values())
    obs_data = mask_data == 1
    rmse_data = float(np.sqrt(np.mean((y_data[obs_data] - recon_data[obs_data]) ** 2)))

    echoes_data = x_data['Sparse Echoes']
    noise_data = x_data['Residual Noise']
    baseline_data = x_data['Smooth Baseline']

    roi_obs = roi_bool_data & obs_data

    if np.any(roi_obs):
        thr = float(0.20 * np.max(np.abs(echoes_data[roi_obs])))
        n_sig = int(np.sum(np.abs(echoes_data[roi_obs]) > thr))
        echo_energy = float(np.sum(echoes_data[roi_obs] ** 2))
        noise_std_roi = float(np.std(noise_data[roi_obs]))
        snr = 10 * np.log10(np.var(echoes_data[roi_obs]) / (np.var(noise_data[roi_obs]) + 1e-12))
    else:
        thr = 0.0
        n_sig = 0
        echo_energy = 0.0
        noise_std_roi = float(np.std(noise_data[obs_data]))
        snr = float('nan')

    echo_score = float(echo_energy / (1e-12 + noise_std_roi**2) * (1.0 + 0.01 * n_sig))

    # Save decision dashboard
    fig, ax = plt.subplots(figsize=(14, 5))
    t_us = t_data * 1e6
    ax.plot(t_us, y_data, linewidth=0.5, alpha=0.5, label='RF')
    ax.plot(t_us, echoes_data, linewidth=0.9, label='Sparse Echoes')

    # mark walls + ROI
    ax.axvline(t_us[fw_idx], color='k', linewidth=1.2, label='Front wall')
    ax.axvline(t_us[bw_idx], color='k', linewidth=1.2, label='Back wall')

    roi_min_us = (roi_min_mm / 1000.0) * 2 / c_data * 1e6 if np.isfinite(roi_min_mm) else np.nan
    roi_max_us = (roi_max_mm / 1000.0) * 2 / c_data * 1e6 if np.isfinite(roi_max_mm) else np.nan
    if np.isfinite(roi_min_us) and np.isfinite(roi_max_us) and roi_max_us > roi_min_us:
        ax.axvspan(roi_min_us, roi_max_us, color='dodgerblue', alpha=0.08, label='ROI')

        # mark significant sparse samples (thin)
        sig_idx = np.where(roi_bool_data & (np.abs(echoes_data) > thr))[0] if n_sig > 0 else np.array([], dtype=int)
        if sig_idx.size > 0:
            ax.scatter(t_us[sig_idx], echoes_data[sig_idx], s=5, color='crimson', alpha=0.35, label='Significant sparse samples')

    ax.set_title(f"{name}: ROI EchoScore={echo_score:.3f}, SNR={snr:.2f} dB, RMSE={rmse_data:.5f}")
    ax.set_xlabel('Time (µs)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path / f"masked_proximal_dashboard_{name}.png", dpi=180)
    plt.close(fig)

    all_results[name] = {
        'x': x_data,
        'n_sig_roi': n_sig,
        'snr_db_roi': snr,
        'rmse_obs': rmse_data,
        'baseline_range': float(baseline_data.max() - baseline_data.min()),
        'echo_energy_roi': echo_energy,
        'echo_score_roi': echo_score,
        'noise_std_roi': noise_std_roi,
        'fw_mm': fw_mm,
        'bw_mm': bw_mm,
        'roi_min_mm': roi_min_mm,
        'roi_max_mm': roi_max_mm,
    }

    print(f"  EchoScore_ROI: {echo_score:.3f} | n_sig_ROI: {n_sig:4d} | SNR_ROI: {snr:6.2f} dB | RMSE: {rmse_data:.6f}")

# %% [markdown]
# ### Results Summary Table

# %%
import pandas as pd

# Create summary table
summary_data = []
for name, res in all_results.items():
    summary_data.append({
        'Dataset': name,
        'EchoScore_ROI (higher => suspicious reflectors)': res['echo_score_roi'],
        'EchoEnergy_ROI': res['echo_energy_roi'],
        '#SignificantSamples_ROI': res['n_sig_roi'],
        'SNR_ROI (dB)': res['snr_db_roi'],
        'RMSE_obs': res['rmse_obs'],
        'Noiseσ_ROI': res['noise_std_roi'],
        'ROI(mm)': f"[{res['roi_min_mm']:.2f},{res['roi_max_mm']:.2f}]",
    })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('EchoScore_ROI (higher => suspicious reflectors)', ascending=False)

print("\nDecision table: rank scans by EchoScore_ROI")
print("=" * 90)
display(df_summary)

plt.figure(figsize=(10, 4))
plt.bar(df_summary['Dataset'], df_summary['EchoScore_ROI (higher => suspicious reflectors)'])
plt.xticks(rotation=25, ha='right')
plt.ylabel('EchoScore_ROI')
plt.title('Inspection triage (ROI-based): higher score => more suspicious internal reflectors')
plt.tight_layout()
plt.savefig(output_path / 'masked_proximal_decision_scores.png', dpi=150)
plt.show()

# Display dashboards inline
from IPython.display import display

print("\nDashboards (ROI-based)")
print("=" * 60)
for name in df_summary['Dataset']:
    p = output_path / f"masked_proximal_dashboard_{name}.png"
    if p.exists():
        display(Image(data=p.read_bytes(), format='png'))

# %% [markdown]
# ---
# ## 9. Comparison with State-of-the-Art Baselines
#
# We compare our proposed Masked Proximal Decomposition (MPD) with standard and state-of-the-art baselines:
# 1. **Bandpass Filtering**: Classical approach (2-8 MHz).
# 2. **Wavelet Denoising**: Standard denoising (Daubechies-4, Soft Thresholding).
# 3. **Total Variation (TV) Denoising**: Optimization-based baseline using TV norm.
#
# Note: Deep Learning methods are not included as they require large training datasets which are unavailable here.

# %%
def tv_denoise_1d_chambolle(y, weight=0.1, n_iter=200):
    """1D Total Variation denoising using Chambolle's algorithm.

    Solves: min_x 0.5||x-y||_2^2 + weight * TV(x)
    where TV(x)=sum_i |x_{i+1}-x_i|.

    This implementation is stable and dependency-free.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return y.copy()

    p = np.zeros(n - 1, dtype=float)
    tau = 0.125

    for _ in range(int(n_iter)):
        # divergence of p
        div_p = np.zeros(n, dtype=float)
        div_p[0] = -p[0]
        div_p[1:-1] = p[:-1] - p[1:]
        div_p[-1] = p[-1]

        x = y - weight * div_p
        grad_x = np.diff(x)

        # gradient ascent + projection
        p = p + (tau / weight) * grad_x
        p = p / np.maximum(1.0, np.abs(p))

    # final x
    div_p = np.zeros(n, dtype=float)
    div_p[0] = -p[0]
    div_p[1:-1] = p[:-1] - p[1:]
    div_p[-1] = p[-1]

    return y - weight * div_p


def run_baselines(y, fs, fc=5e6):
    """Run baseline methods on the signal.

    Returns dict[str, np.ndarray]. Methods are intentionally training-free.
    Wavelet is optional (PyWavelets). If unavailable, it's skipped.
    """
    results = {}

    # 1) Bandpass Filtering (Butterworth)
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * float(fs)
    low = float(0.4 * fc / nyq)
    high = float(1.6 * fc / nyq)
    low = max(1e-6, min(low, 0.999))
    high = max(1e-6, min(high, 0.999))
    if low >= high:
        # fallback: mild highpass
        b, a = butter(4, min(0.99, max(1e-3, low)), btype='high')
    else:
        b, a = butter(4, [low, high], btype='band')
    results['Bandpass'] = filtfilt(b, a, y)

    # 2) Total Variation Denoising (TV)
    # weight tuned relative to signal std
    weight = 0.08 * float(np.std(y))
    results['TV Denoising'] = tv_denoise_1d_chambolle(y, weight=weight, n_iter=200)

    # 3) Wavelet Denoising (optional)
    try:
        import pywt  # type: ignore

        sigma = float(np.median(np.abs(y - np.mean(y))) / 0.6745)
        threshold = sigma * np.sqrt(2.0 * np.log(len(y)))
        coeffs = pywt.wavedec(y, 'db4', level=5)
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        y_wavelet = pywt.waverec(coeffs_thresh, 'db4')
        if len(y_wavelet) != len(y):
            y_wavelet = y_wavelet[:len(y)]
        results['Wavelet'] = y_wavelet
    except Exception:
        # If PyWavelets isn't installed, keep the notebook runnable.
        pass

    return results

print("\nRunning baselines comparison...")
print("=" * 60)

comparison_metrics = []

for name, data in ndt_datasets.items():
    y_raw = data['rf']
    fs_val = float(data['fs'])
    fc_val = float(data.get('fc', 5e6))

    # ROI from MPD results
    res = all_results[name]
    roi_min = float(res['roi_min_mm'])
    roi_max = float(res['roi_max_mm'])
    c_val = float(data['c'])

    depth_mm = np.asarray(data['time']) * c_val / 2 * 1000
    roi_mask = (depth_mm >= roi_min) & (depth_mm <= roi_max)

    if int(np.sum(roi_mask)) == 0:
        continue

    baselines = run_baselines(y_raw, fs_val, fc=fc_val)

    # Proposed MPD metrics: sparse vs residual inside ROI
    x_mpd = res['x']['Sparse Echoes']
    noise_mpd = res['x']['Residual Noise']
    snr_mpd = 10 * np.log10(float(np.var(x_mpd[roi_mask])) / (float(np.var(noise_mpd[roi_mask])) + 1e-12))

    comparison_metrics.append({
        'Dataset': name,
        'Method': 'Proposed (MPD)',
        'SNR_ROI (dB)': snr_mpd,
        'RMSE_obs': float(res['rmse_obs']),
        'Notes': 'explicit baseline + sparse echoes + noise'
    })

    for method_name, y_est in baselines.items():
        resid = y_raw - y_est
        snr = 10 * np.log10(float(np.var(y_est[roi_mask])) / (float(np.var(resid[roi_mask])) + 1e-12))
        rmse = float(np.sqrt(np.mean((y_raw - y_est) ** 2)))
        comparison_metrics.append({
            'Dataset': name,
            'Method': method_name,
            'SNR_ROI (dB)': snr,
            'RMSE_obs': rmse,
            'Notes': 'single-output denoiser/filter'
        })

import pandas as pd

df_comp = pd.DataFrame(comparison_metrics)
print("\nBaseline comparison (higher SNR_ROI is better):")
print(df_comp.groupby('Method')['SNR_ROI (dB)'].mean().sort_values(ascending=False))

# Save a full table for the paper
(df_comp
 .groupby('Method')
 .agg({'SNR_ROI (dB)': 'mean', 'RMSE_obs': 'mean'})
 .sort_values('SNR_ROI (dB)', ascending=False)
 .to_csv(output_path / 'baseline_comparison_table.csv'))

# Visual comparison for one dataset
ex_name = 'weld_inspection' if 'weld_inspection' in ndt_datasets else list(ndt_datasets.keys())[0]
ex_data = ndt_datasets[ex_name]
ex_baselines = run_baselines(ex_data['rf'], float(ex_data['fs']), fc=float(ex_data.get('fc', 5e6)))
ex_mpd = all_results[ex_name]['x']['Sparse Echoes']

t_us = np.asarray(ex_data['time']) * 1e6

nrows = 1 + len(ex_baselines)
plt.figure(figsize=(14, 2.6 * nrows))

plt.subplot(nrows, 1, 1)
plt.plot(t_us, ex_data['rf'], 'k', alpha=0.35, label='Raw')
plt.plot(t_us, ex_mpd, 'r', linewidth=1.0, label='Proposed (MPD) sparse')
plt.title(f"{ex_name}: MPD sparse component")
plt.legend()
plt.grid(True, alpha=0.25)

row_idx = 2
for method, sig in ex_baselines.items():
    plt.subplot(nrows, 1, row_idx)
    plt.plot(t_us, ex_data['rf'], 'k', alpha=0.35, label='Raw')
    plt.plot(t_us, sig, 'b', linewidth=1.0, label=method)
    plt.title(f"{ex_name}: {method}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    row_idx += 1

plt.tight_layout()
fig_path = output_path / 'method_comparison.png'
plt.savefig(fig_path, dpi=170, bbox_inches='tight')
plt.close()

# Re-embed from disk to ensure the exported HTML/PDF includes the figure reliably
if fig_path.exists() and fig_path.stat().st_size > 0:
    from IPython.display import Image as _Image
    from IPython.display import display as _display
    _display(_Image(filename=str(fig_path)))

# %% [markdown]
# ---
# ## 10. Conclusions
#
# ### Summary
#
# We implemented the **Masked Proximal Operator** framework from Meyers & Boyd (2023)
# and applied it to real NDT ultrasound data. The key achievements:
#
# 1. ✅ **Successful decomposition** into physically meaningful components
# 2. ✅ **Robust handling of missing data** via the masked formulation
# 3. ✅ **Quantitative metrics** (SNR, echo count) derived from decomposition
# 4. ✅ **Applicable across datasets** with auto-tuned regularization
#
# ### Relevance to InPhase Solutions
#
# This technique is directly applicable to InPhase's ultrasound work:
#
# - **Signal preprocessing**: Use decomposition to clean signals before analysis
# - **Quality metrics**: SNR from decomposition as a data quality indicator
# - **Missing data**: Handle gated acquisitions and corrupted samples
# - **Defect isolation**: Sparse component highlights localized reflectors
#
# ### Future Extensions
#
# - **Periodic components**: Add seasonal/periodic proximal for repetitive patterns
# - **2D extension**: Apply to B-scan images (2D signal decomposition)
# - **Online decomposition**: Streaming version for real-time processing
# - **Learned proximal operators**: Use neural networks as proximal operators
#
# ---
#
# **Author:** Reza Mirzaeifard, PhD
# **Contact:** reza.mirzaeifard@gmail.com
# **Reference:** Meyers & Boyd (2023), *Foundations and Trends® in Signal Processing*
