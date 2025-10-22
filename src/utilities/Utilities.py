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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-10-20
:Description: miscellaneous functions
"""


# ============================================================
# Imports
# ============================================================

import jax
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
        Convergence tolerance. Iteration stops when:
        |x_new - x_old| < tol * (1 + |x_old|) or |f(x)| < tol
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
    - Uses relative tolerance for better handling of roots at different scales
    - The algorithm is JIT-compiled for performance

    """

    # Convert to complex arithmetic to handle complex roots
    coeffs = coeffs.astype(jnp.complex128)
    x = jnp.asarray(init_guess, dtype=jnp.complex128)

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
        converged_value = jnp.abs(f_x) < tol
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
