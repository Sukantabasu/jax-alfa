# Copyright (C) 2025 Sukanta Basu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
File: Utilities.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-10-20
:Description: miscellaneous functions
"""


# ============================================================
# Imports
# ============================================================

import jax
import numpy as np
import jax.numpy as jnp


# ============================================================
#  Compute planar averaged values of a 3D field
# ============================================================

@jax.jit
def PlanarMean(F):
    """
    Computes horizontal average of a 3D field at each vertical level.

    Parameters:
    -----------
    F : jax.numpy.ndarray
        3D array with shape (nx, ny, nz)

    Returns:
    --------
    jax.numpy.ndarray
        1D array of length nz containing planar-averaged values
    """

    return jnp.mean(F, axis=(0, 1))


# ============================================================
#  Compute averaging of a 3D array along the z-direction
# ============================================================

@jax.jit
def StagGridAvg(F):
    """
    Computes averaging of a 3D array along the z-direction.

    Parameters:
    -----------
    F : jax.numpy.ndarray
        3D array with shape (nx, ny, nz)

    Returns:
    --------
    jax.numpy.ndarray
        Averaged field with shape (nx, ny, nz-1)
    """

    return 0.5 * (F[:, :, :-1] + F[:, :, 1:])


# ============================================================
# Compute moving average filtering of 2D fields
# ============================================================

@jax.jit
def Imfilter(x):
    """
    Applies a 3x3 box filter to a 2D field with periodic boundaries.

    Parameters:
    -----------
    x : jax.numpy.ndarray
        2D array representing a horizontal slice

    Returns:
    --------
    jax.numpy.ndarray
        Filtered 2D array with the same shape as input
    """

    kernel = jnp.ones((3, 3)) / 9.0  # 3x3 mean filter (box filter)

    # Apply periodic padding (wrap around)
    x_padded = jnp.pad(x, ((1, 1), (1, 1)), mode='wrap')

    # Perform 2D convolution
    x_filtered = jax.lax.conv_general_dilated(
        x_padded[None, None, :, :],  # Add batch & channel dims
        kernel[None, None, :, :],    # Kernel shape: [Out_C, In_C, H, W]
        (1, 1),  # Stride (1,1) ensures all pixels are processed
        'VALID'  # No extra padding applied beyond what we added
    )

    return x_filtered[0, 0, :, :]  # Remove batch & channel dims


# ============================================================
# Scale-dependence parameter from the LASDD polynomial
# ============================================================

def PolyRootsAberth(C, n_iter=60, max_radius=1e4):
    """
    All roots of real polynomials, one per row of C (nz, n+1; descending
    order), by the Aberth-Ehrlich simultaneous iteration in the working
    precision on the device (vectorized over rows; no deflation, no host
    callback, fixed iteration count).

    Returns:
    --------
    z : complex ndarray (nz, n)
        Roots.
    ok : bool ndarray (nz,)
        False if the row is non-finite or zero, has a root bound above
        max_radius (a huge root), or has not converged (backward error
        |P(z)| > 64 n eps sum_i |c_i||z|^(n-i) for some root).
    """
    nlev, ncoef = C.shape
    n = ncoef - 1
    eps = jnp.finfo(C.dtype).eps
    cdt = jnp.complex128 if C.dtype == jnp.float64 else jnp.complex64

    # root bound max_i |c_i/c_0|^(1/i); rows without a usable bound are skipped
    finite = jnp.all(jnp.isfinite(C), axis=1) & (C[:, 0] != 0)
    c0 = jnp.where(finite, C[:, 0], 1.0)
    Cm = jnp.where(finite[:, None], C / c0[:, None], 0.0)      # monic
    rad = jnp.max(jnp.abs(Cm[:, 1:]) ** (1.0 / jnp.arange(1, n + 1)), axis=1)
    ok = finite & jnp.isfinite(rad) & (rad <= max_radius)
    rad = jnp.where(ok & (rad > 0), rad, 1.0)
    Cm = jnp.where(ok[:, None], Cm, jnp.zeros(ncoef).at[0].set(1.0).at[n].set(-1.0))

    # start on a circle rotated off the real axis
    ang = 2 * np.pi * np.arange(n) / n + 0.4
    z0 = (rad[:, None] * jnp.exp(1j * jnp.asarray(ang))[None, :]).astype(cdt)
    eye = jnp.eye(n, dtype=bool)[None]
    c = Cm.astype(cdt)[:, None, :]

    def horner(z):
        p = jnp.broadcast_to(c[..., 0], z.shape)
        dp = jnp.zeros_like(z)
        for k in range(1, n + 1):
            dp = dp * z + p
            p = p * z + c[..., k]
        return p, dp

    def body(_, z):
        p, dp = horner(z)
        w = p / dp
        d = z[:, :, None] - z[:, None, :]
        S = jnp.sum(jnp.where(eye, 0.0, 1.0 / jnp.where(eye, 1.0, d)), axis=2)
        zn = z - w / (1.0 - w * S)
        return jnp.where(jnp.isfinite(zn), zn, z)

    z = jax.lax.fori_loop(0, n_iter, body, z0)

    p, _ = horner(z)
    az = jnp.abs(z)
    bnd = jnp.zeros_like(az)
    for k in range(n + 1):
        bnd = bnd * az + jnp.abs(Cm[:, None, k])
    ok = ok & jnp.all(jnp.abs(p) <= 64 * n * eps * bnd, axis=1)
    return z, ok


@jax.jit
def MaxRealRootBeta(C):
    """
    Scale-dependence parameter beta(z) from the fifth-order LASDD polynomials,
    one per vertical level, with the selection rule of the original MATLAB
    code (MLp_LocSMAG.m, MLp_LocScSMAG.m): beta is the largest real root
    (complex roots count as 0), and beta = 1 (scale invariance) if that root
    is not in (0, 5).

    Simple and conservative (owner's choice, 2026-11-19): beta = 1 also when
    the roots are not computed reliably -- no convergence, a huge root, any
    root that is neither clearly real (|Im z| <= 1000 eps |z|) nor clearly
    complex (|Im z| > 0.01 |z|), or (single precision only) two roots closer
    than 5% of their size, where float32 can split or merge them. In tests,
    LES polynomials had well-separated roots (min relative distance 0.1).

    Parameters:
    -----------
    C : ndarray of shape (nz, 6)
        Polynomial coefficients per level, descending order of degree.

    Returns:
    --------
    beta : ndarray of shape (nz,)
    """
    z, ok = PolyRootsAberth(C)
    eps = jnp.finfo(C.dtype).eps
    rel = jnp.abs(z.imag) / jnp.maximum(jnp.abs(z), jnp.finfo(C.dtype).tiny)
    real = rel <= 1000 * eps
    doubtful = jnp.any(~real & (rel <= 0.01), axis=1)
    if C.dtype == jnp.float32:
        n = z.shape[1]
        dist = jnp.abs(z[:, :, None] - z[:, None, :])
        size = jnp.maximum(jnp.abs(z[:, :, None]), jnp.abs(z[:, None, :]))
        close = (dist <= 0.05 * size) & ~jnp.eye(n, dtype=bool)[None]
        doubtful = doubtful | jnp.any(close, axis=(1, 2))
    bmax = jnp.max(jnp.where(real, z.real, 0.0), axis=1)
    valid = ok & ~doubtful & (bmax > 0.0) & (bmax < 5.0)
    return jnp.where(valid, bmax, 1.0).astype(C.dtype)


# ============================================================
# Compute roots of a polynomial using Laguerre's method
# ============================================================

@jax.jit
def Roots(coeffs, init_guess=1.0 + 0j, tol=1e-6, max_iter=20):
    """
    Find a root of a polynomial using Laguerre's method.

    Laguerre's method is a root-finding algorithm that works well for
    polynomials and converges cubically for simple roots. It works in
    the complex plane and can find complex roots.

    Parameters:
    -----------
    coeffs : ndarray
        Polynomial coefficients in descending order of degree.
        For a 5th degree polynomial: [a5, a4, a3, a2, a1, a0]
        represents: a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    init_guess : complex, optional
        Initial guess for the root. Default: 1.0+0j
        For real polynomials, can use real guess, but complex arithmetic
        is used internally to handle complex roots.
    tol : float, optional
        Relative step tolerance. Iteration stops when
        |x_new - x_old| < tol * (1 + |x_old|), or when the residual of the
        normalized polynomial is at round-off level (|f(x)| < 10*eps).
        Default: 1e-6
    max_iter : int, optional
        Maximum number of iterations. Default: 20

    Returns:
    --------
    complex
        Root of the polynomial if converged, or nan+0j if not converged
        within max_iter iterations.

    Notes:
    ------
    - Uses complex arithmetic internally to handle complex roots
    - Includes numerical safeguards against division by zero
    - The coefficients are normalized by their largest magnitude, so the
      result does not depend on the overall scale of the polynomial (the
      dynamic-model coefficients scale with powers of the resolved fields).
      Before 2026-11-19 an absolute residual test |f(x)| < tol made the root
      depend on field amplitude.
    - The algorithm is JIT-compiled for performance

    """

    # Convert to complex arithmetic to handle complex roots
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    rdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    coeffs = coeffs.astype(cdtype)
    # Normalize: roots are unchanged, the residual test becomes scale-free
    scale = jnp.max(jnp.abs(coeffs))
    coeffs = coeffs / jnp.where(scale > 0, scale, 1.0)
    res_tol = 10 * jnp.finfo(rdtype).eps
    x = jnp.asarray(init_guess, dtype=cdtype)

    # Polynomial degree
    n = coeffs.shape[0] - 1

    # Helper functions for polynomial evaluation
    def polynomial(p, x):
        """Evaluate polynomial at x using Horner's method."""
        return jnp.polyval(p, x)

    def derivative(p, x):
        """Evaluate first derivative at x."""
        dp = jnp.polyder(p)
        return jnp.polyval(dp, x)

    def second_derivative(p, x):
        """Evaluate second derivative at x."""
        d2p = jnp.polyder(jnp.polyder(p))
        return jnp.polyval(d2p, x)

    # Loop continuation condition
    def cond_fn(state):
        x, iteration, converged = state
        return (iteration < max_iter) & (~converged)

    # Laguerre iteration step
    def body_fn(state):
        x, iteration, _ = state

        # Evaluate polynomial and derivatives
        f_x = polynomial(coeffs, x)
        df_x = derivative(coeffs, x)
        d2f_x = second_derivative(coeffs, x)

        # Numerical safeguard: small epsilon for complex arithmetic
        eps = 1e-16 + 0j

        # Protect against division by zero
        f_safe = jnp.where(jnp.abs(f_x) < eps, eps, f_x)

        # Laguerre's method formulas
        G = df_x / f_safe
        H = G ** 2 - d2f_x / f_safe

        # Compute discriminant: (n-1) * (n*H - G²)
        # This is the CORRECTED formula
        discriminant = (n - 1) * (n * H - G ** 2)
        sqrt_disc = jnp.sqrt(discriminant)

        # Choose denominator with larger absolute value for stability
        denom1 = G + sqrt_disc
        denom2 = G - sqrt_disc
        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        # Protect against division by zero
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)

        # Laguerre update step
        x_new = x - n / denom

        # Check convergence using both relative step size and function value
        # This handles both large and small roots effectively
        converged_step = jnp.abs(x_new - x) < tol * (1 + jnp.abs(x))
        converged_value = jnp.abs(f_x) < res_tol
        new_converged = converged_step | converged_value

        return x_new, iteration + 1, new_converged

    # Initialize state and run iteration loop
    init_state = (x, jnp.array(0), jnp.array(False))
    final_x, final_iter, converged = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    # Return root if converged, otherwise return nan
    return jnp.where(converged, final_x, jnp.nan + 0j)


# ============================================================
#  Measure memory usage
# ============================================================

@jax.jit
def LogMemory():
    """
    Prints memory usage statistics for all available JAX devices.
    Converts memory values to MB for readability.
    """

    devices = jax.devices()
    for device in devices:
        print(f"\nDevice: {device}")
        stats = device.memory_stats()

        if stats is None:
            print("Memory stats not available for this device")
            continue

        # Convert to MB for readability
        bytes_in_use = stats.get('bytes_in_use', 0) / (1024 * 1024)
        peak_bytes = stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
        allocated_bytes = stats.get('bytes_allocated', 0) / (1024 * 1024)

        print(f"Current memory usage: {bytes_in_use:.2f} MB")
        print(f"Peak memory usage: {peak_bytes:.2f} MB")
        print(f"Allocated memory: {allocated_bytes:.2f} MB")

        # Print all available stats for debugging
        print("\nAll available memory stats:")
        for key, value in stats.items():
            print(f"{key}: {value / (1024 * 1024):.2f} MB")
