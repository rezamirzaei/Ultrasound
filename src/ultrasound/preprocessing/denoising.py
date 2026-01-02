"""
Advanced Denoising Methods for Ultrasound Images.

This module implements optimization-based denoising methods including:
- Total Variation (TV) denoising
- ADMM-based TV denoising (showcasing optimization expertise)
- Bilateral filtering

These methods are particularly effective for ultrasound images as they
preserve edges while removing noise.

The ADMM implementation demonstrates expertise in convex optimization
and the Alternating Direction Method of Multipliers, which is central
to the PhD research on non-convex optimization.

References:
    Rudin, L.I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation 
    based noise removal algorithms. Physica D, 60(1-4), 259-268.
    
    Boyd, S., et al. (2011). Distributed optimization and statistical 
    learning via the alternating direction method of multipliers.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import diags, vstack, csr_matrix
from scipy.sparse.linalg import cg
import cv2
from typing import Optional, Tuple


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
    
    This preserves edges while smoothing homogeneous regions.
    
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
        # Compute gradients
        grad_x = np.roll(u, -1, axis=1) - u
        grad_y = np.roll(u, -1, axis=0) - u
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + eps**2)
        
        # Normalized gradients
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        
        # Divergence of normalized gradients
        div = (np.roll(nx, 1, axis=1) - nx) + (np.roll(ny, 1, axis=0) - ny)
        
        # Update u
        u = img + weight * div
        
    return np.clip(u * 255, 0, 255).astype(np.uint8)


def admm_tv_denoising(
    image: np.ndarray,
    lambda_tv: float = 0.1,
    rho: float = 1.0,
    n_iter: int = 50,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Total Variation denoising using ADMM (Alternating Direction Method of Multipliers).
    
    This implementation showcases expertise in ADMM optimization, which is
    central to my PhD research on non-convex and non-smooth optimization.
    
    Problem formulation:
        min_u  (1/2)||u - f||^2 + lambda * ||Du||_1
    
    ADMM reformulation:
        min_u,z  (1/2)||u - f||^2 + lambda * ||z||_1
        subject to: Du = z
    
    The augmented Lagrangian is:
        L(u, z, y) = (1/2)||u - f||^2 + lambda*||z||_1 + y^T(Du - z) + (rho/2)||Du - z||^2
    
    ADMM iterations:
        u^{k+1} = argmin_u L(u, z^k, y^k)  [u-update: linear system]
        z^{k+1} = argmin_z L(u^{k+1}, z, y^k)  [z-update: soft thresholding]
        y^{k+1} = y^k + rho * (Du^{k+1} - z^{k+1})  [dual update]
    
    Args:
        image: Input noisy image
        lambda_tv: TV regularization weight
        rho: ADMM penalty parameter
        n_iter: Number of ADMM iterations
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
    z = np.zeros(2 * N)  # [z_x; z_y] for horizontal and vertical gradients
    y = np.zeros(2 * N)  # Dual variables
    
    # Build difference operators
    Dx, Dy = _build_difference_operators(m, n)
    D = vstack([Dx, Dy])  # Stack horizontal and vertical difference operators
    DTD = D.T @ D
    
    # Precompute for u-update (I + rho * D^T D)
    I = diags([1.0] * N, format='csr')
    A = I + rho * DTD
    
    # Convergence tracking
    primal_residuals = []
    dual_residuals = []
    
    for k in range(n_iter):
        u_old = u.copy()
        
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
        r = Du - z  # Primal residual
        y = y + rho * r
        
        # ============ Convergence check ============
        primal_res = np.linalg.norm(r)
        dual_res = np.linalg.norm(rho * D.T @ (z - z_tilde))
        
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)
        
        if verbose and (k + 1) % 10 == 0:
            print(f"Iter {k+1}: primal_res={primal_res:.6f}, dual_res={dual_res:.6f}")
    
    # Reshape and convert back
    u_image = u.reshape(m, n)
    u_image = np.clip(u_image * 255, 0, 255).astype(np.uint8)
    
    convergence_info = {
        'primal_residuals': primal_residuals,
        'dual_residuals': dual_residuals,
        'n_iter': n_iter,
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
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def _build_difference_operators(m: int, n: int) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build sparse finite difference operators for 2D image.
    
    Creates operators Dx and Dy for computing horizontal and vertical
    differences with periodic boundary conditions.
    
    Args:
        m: Number of rows
        n: Number of columns
        
    Returns:
        Tuple of (Dx, Dy) sparse matrices
    """
    N = m * n
    
    # Horizontal differences (Dx)
    # d/dx[i,j] = u[i,j+1] - u[i,j]
    row_idx = []
    col_idx = []
    data = []
    
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            idx_right = i * n + (j + 1) % n  # Periodic boundary
            
            row_idx.extend([idx, idx])
            col_idx.extend([idx, idx_right])
            data.extend([-1, 1])
    
    Dx = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    
    # Vertical differences (Dy)
    # d/dy[i,j] = u[i+1,j] - u[i,j]
    row_idx = []
    col_idx = []
    data = []
    
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            idx_below = ((i + 1) % m) * n + j  # Periodic boundary
            
            row_idx.extend([idx, idx])
            col_idx.extend([idx, idx_below])
            data.extend([-1, 1])
    
    Dy = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    
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
        # Compute gradients in 4 directions
        nabla_n = np.roll(img, -1, axis=0) - img  # North
        nabla_s = np.roll(img, 1, axis=0) - img   # South
        nabla_e = np.roll(img, -1, axis=1) - img  # East
        nabla_w = np.roll(img, 1, axis=1) - img   # West
        
        # Edge-stopping function
        if option == 1:
            # Exponential: favors high-contrast edges
            c_n = np.exp(-(nabla_n / kappa) ** 2)
            c_s = np.exp(-(nabla_s / kappa) ** 2)
            c_e = np.exp(-(nabla_e / kappa) ** 2)
            c_w = np.exp(-(nabla_w / kappa) ** 2)
        else:
            # Quadratic: favors wide regions over smaller ones
            c_n = 1 / (1 + (nabla_n / kappa) ** 2)
            c_s = 1 / (1 + (nabla_s / kappa) ** 2)
            c_e = 1 / (1 + (nabla_e / kappa) ** 2)
            c_w = 1 / (1 + (nabla_w / kappa) ** 2)
        
        # Update image
        img = img + gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)
    
    return np.clip(img, 0, 255).astype(np.uint8)
