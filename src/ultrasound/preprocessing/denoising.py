"""
Advanced Denoising Methods for Ultrasound Images.

This module implements optimization-based denoising methods including:
- Total Variation (TV) denoising
- ADMM-based TV denoising with adaptive penalty
- Bilateral filtering
- Anisotropic diffusion

These methods are particularly effective for ultrasound images as they
preserve edges while removing noise.

References:
    Rudin, L.I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation
    based noise removal algorithms. Physica D, 60(1-4), 259-268.

    Boyd, S., et al. (2011). Distributed optimization and statistical
    learning via the alternating direction method of multipliers.
"""

from typing import cast

import cv2
import numpy as np
from scipy.sparse import csr_matrix, eye, kron, lil_matrix, vstack
from scipy.sparse.linalg import cg


def total_variation_denoising(
    image: np.ndarray,
    weight: float = 0.1,
    n_iter: int = 100,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Total Variation denoising using gradient descent.

    Minimizes the objective:
        min_u  ||u - f||^2 + weight * TV(u)

    where TV(u) is the total variation (sum of gradient magnitudes).

    Uses reflexive (Neumann) boundary conditions to avoid edge artifacts.

    Args:
        image: Input noisy image
        weight: Regularization weight (higher = more smoothing)
        n_iter: Number of iterations
        eps: Small constant for numerical stability

    Returns:
        Denoised image
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    u = img.copy()

    for _ in range(n_iter):
        # Compute gradients with reflexive (zero-flux) boundary conditions
        grad_x = np.zeros_like(u)
        grad_y = np.zeros_like(u)
        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]  # Zero at right boundary
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]  # Zero at bottom boundary

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + eps**2)

        # Normalized gradients
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag

        # Divergence with matching boundary conditions
        div_x = np.zeros_like(u)
        div_y = np.zeros_like(u)
        div_x[:, 1:-1] = nx[:, 1:-1] - nx[:, :-2]
        div_x[:, 0] = nx[:, 0]
        div_x[:, -1] = -nx[:, -2]
        div_y[1:-1, :] = ny[1:-1, :] - ny[:-2, :]
        div_y[0, :] = ny[0, :]
        div_y[-1, :] = -ny[-2, :]

        div = div_x + div_y

        # Update u
        u = img + weight * div

    return np.clip(u * 255, 0, 255).astype(np.uint8)


def admm_tv_denoising(
    image: np.ndarray,
    lambda_tv: float = 0.1,
    rho: float = 1.0,
    n_iter: int = 50,
    adaptive_rho: bool = True,
    mu: float = 10.0,
    tau_incr: float = 2.0,
    tau_decr: float = 2.0,
    abstol: float = 1e-4,
    reltol: float = 1e-3,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Total Variation denoising using ADMM (Alternating Direction Method of Multipliers).

    Problem formulation:
        min_u  (1/2)||u - f||^2 + lambda * ||Du||_1

    ADMM reformulation:
        min_u,z  (1/2)||u - f||^2 + lambda * ||z||_1
        subject to: Du = z

    ADMM iterations:
        u^{k+1} = argmin_u L(u, z^k, y^k)  [u-update: linear system]
        z^{k+1} = argmin_z L(u^{k+1}, z, y^k)  [z-update: soft thresholding]
        y^{k+1} = y^k + rho * (Du^{k+1} - z^{k+1})  [dual update]

    Features:
        - Reflexive boundary conditions (no edge artifacts)
        - Vectorised sparse operator construction (fast)
        - Adaptive penalty parameter rho (Boyd et al., §3.4.1)
        - Early stopping based on primal/dual residual tolerances

    Args:
        image: Input noisy image (2-D, uint8 or float)
        lambda_tv: TV regularization weight
        rho: Initial ADMM penalty parameter
        n_iter: Maximum number of ADMM iterations
        adaptive_rho: Enable adaptive rho update (Boyd et al., 2011, §3.4.1)
        mu: Threshold ratio for adaptive rho balancing
        tau_incr: Factor to increase rho when primal residual is too large
        tau_decr: Factor to decrease rho when dual residual is too large
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        verbose: Print convergence information

    Returns:
        Tuple of (denoised_image, convergence_info)
    """
    # Normalize input
    f = image.astype(np.float64)
    if f.max() > 1:
        f = f / 255.0

    m, n = f.shape
    N = m * n

    # Flatten image
    f_vec = f.flatten()

    # Initialize variables
    u = f_vec.copy()
    z = np.zeros(2 * N)  # [z_x; z_y]
    y = np.zeros(2 * N)  # Dual variables

    # Build difference operators (vectorised, reflexive boundaries)
    Dx, Dy = _build_difference_operators(m, n)
    D = vstack([Dx, Dy])
    DTD = D.T @ D

    # Precompute for u-update: (I + rho * D^T D)
    I_sp = eye(N, format="csr")

    def _build_A(rho_val):
        return I_sp + rho_val * DTD

    A = _build_A(rho)

    # Convergence tracking
    primal_residuals = []
    dual_residuals = []
    eps_pri_list = []
    eps_dual_list = []
    rho_history = [rho]

    sqrt_n = np.sqrt(2 * N)
    sqrt_p = np.sqrt(N)

    for k in range(n_iter):
        z_old = z.copy()

        # ============ u-update ============
        # Solve: (I + rho * D^T D) u = f + D^T (rho * z - y)
        rhs = f_vec + D.T @ (rho * z - y)
        u, _ = cg(A, rhs, x0=u, maxiter=50)

        # ============ z-update (soft thresholding) ============
        # z = S_{lambda/rho}(Du + y/rho)
        Du = D @ u
        z_tilde = Du + y / rho
        z = soft_threshold(z_tilde, lambda_tv / rho)

        # ============ y-update (dual ascent) ============
        r = Du - z  # Primal residual vector
        y = y + rho * r

        # ============ Convergence diagnostics ============
        # Dual residual: rho * D^T * (z^{k+1} - z^k)
        s = rho * D.T @ (z - z_old)

        primal_res = np.linalg.norm(r)
        dual_res = np.linalg.norm(s)

        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)

        # Tolerances (Boyd et al., §3.3.1)
        eps_pri = float(sqrt_n * abstol + reltol * max(np.linalg.norm(Du), np.linalg.norm(z)))
        eps_dual = float(sqrt_p * abstol + reltol * np.linalg.norm(D.T @ y))
        eps_pri_list.append(eps_pri)
        eps_dual_list.append(eps_dual)

        if verbose and (k + 1) % 10 == 0:
            print(
                f"Iter {k+1}: primal_res={primal_res:.6f} (eps={eps_pri:.6f}), "
                f"dual_res={dual_res:.6f} (eps={eps_dual:.6f}), rho={rho:.4f}"
            )

        # Early stopping
        if primal_res < eps_pri and dual_res < eps_dual:
            if verbose:
                print(f"Converged at iteration {k+1}")
            break

        # ============ Adaptive rho (Boyd et al., §3.4.1) ============
        if adaptive_rho:
            if primal_res > mu * dual_res:
                rho *= tau_incr
                y /= tau_incr  # Scale dual variable to keep y/rho constant
                A = _build_A(rho)
            elif dual_res > mu * primal_res:
                rho /= tau_decr
                y *= tau_decr
                A = _build_A(rho)
            rho_history.append(rho)

    # Reshape and convert back
    u_image = u.reshape(m, n)
    u_image = np.clip(u_image * 255, 0, 255).astype(np.uint8)

    convergence_info = {
        "primal_residuals": primal_residuals,
        "dual_residuals": dual_residuals,
        "eps_pri": eps_pri_list,
        "eps_dual": eps_dual_list,
        "rho_history": rho_history,
        "n_iter": len(primal_residuals),
    }

    return u_image, convergence_info


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding (proximal operator for L1 norm).

    S_t(x) = sign(x) * max(|x| - t, 0)

    This is the closed-form solution to:
        min_z (1/2)||z - x||^2 + t * ||z||_1

    Args:
        x: Input array
        threshold: Threshold value

    Returns:
        Soft-thresholded array
    """
    return cast(np.ndarray, np.sign(x) * np.maximum(np.abs(x) - threshold, 0))


def _build_difference_operators(m: int, n: int) -> tuple[csr_matrix, csr_matrix]:
    """
    Build sparse finite difference operators for a 2-D image.

    Uses reflexive (Neumann) boundary conditions:  the gradient at the
    boundary is set to zero, which avoids artificial wrap-around artefacts
    that periodic boundaries would create on natural / medical images.

    Construction is fully vectorised via Kronecker products — no Python
    loops over pixels.

    Args:
        m: Number of rows
        n: Number of columns

    Returns:
        Tuple of (Dx, Dy) sparse matrices of shape (m*n, m*n)
    """

    # --- 1-D forward difference with zero boundary (size k) ---
    def _diff1d(k: int) -> csr_matrix:
        # D1 u = [u1-u0, u2-u1, ..., u_{k-1}-u_{k-2}, 0]
        # Last row is zero (reflexive / Neumann boundary)

        d = lil_matrix((k, k))
        for i in range(k - 1):
            d[i, i] = -1.0
            d[i, i + 1] = 1.0
        return d.tocsr()

    # Horizontal differences: Dx = I_m ⊗ D_n
    Dx = kron(eye(m, format="csr"), _diff1d(n), format="csr")

    # Vertical differences: Dy = D_m ⊗ I_n
    Dy = kron(_diff1d(m), eye(n, format="csr"), format="csr")

    return Dx, Dy


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Bilateral filtering for edge-preserving smoothing.

    The bilateral filter averages pixels based on both spatial proximity
    and intensity similarity, preserving edges while smoothing homogeneous regions.

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space (larger = more color mixing)
        sigma_space: Filter sigma in coordinate space (larger = more spatial smoothing)

    Returns:
        Filtered image
    """
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def anisotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 10,
    kappa: float = 50,
    gamma: float = 0.1,
    option: int = 1,
) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion for edge-preserving smoothing.

    Implements the diffusion equation:
        ∂I/∂t = div(c(x,y,t) ∇I)

    where c is an edge-stopping function.

    Args:
        image: Input image
        n_iter: Number of iterations
        kappa: Conduction coefficient (controls sensitivity to edges)
        gamma: Integration constant (time step, should be <= 0.25)
        option: Diffusion function (1: exponential, 2: quadratic)

    Returns:
        Diffused image
    """
    img = image.astype(np.float64)

    for _ in range(n_iter):
        # Compute gradients in 4 directions using zero-padded differences
        nabla_n = np.zeros_like(img)
        nabla_s = np.zeros_like(img)
        nabla_e = np.zeros_like(img)
        nabla_w = np.zeros_like(img)
        nabla_n[:-1, :] = img[1:, :] - img[:-1, :]
        nabla_s[1:, :] = img[:-1, :] - img[1:, :]
        nabla_e[:, :-1] = img[:, 1:] - img[:, :-1]
        nabla_w[:, 1:] = img[:, :-1] - img[:, 1:]

        # Edge-stopping function
        if option == 1:
            # Exponential: favors high-contrast edges
            c_n = np.exp(-((nabla_n / kappa) ** 2))
            c_s = np.exp(-((nabla_s / kappa) ** 2))
            c_e = np.exp(-((nabla_e / kappa) ** 2))
            c_w = np.exp(-((nabla_w / kappa) ** 2))
        else:
            # Quadratic: favors wide regions over smaller ones
            c_n = 1 / (1 + (nabla_n / kappa) ** 2)
            c_s = 1 / (1 + (nabla_s / kappa) ** 2)
            c_e = 1 / (1 + (nabla_e / kappa) ** 2)
            c_w = 1 / (1 + (nabla_w / kappa) ** 2)

        # Update image
        img = img + gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)

    return np.clip(img, 0, 255).astype(np.uint8)
