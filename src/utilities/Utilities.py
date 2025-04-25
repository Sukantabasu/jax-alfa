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
:Date: 2025-4-7
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
def Roots(coeffs, init_guess=1.0, tol=1e-6, max_iter=20):
    """
    Finds a root of a polynomial using Laguerre's method.

    Parameters:
    -----------
    coeffs : jnp.ndarray
        Polynomial coefficients (highest degree first)
    init_guess : float
        Initial guess, default=1.0
    tol : float
        Convergence tolerance, default=1e-6
    max_iter : int
        Maximum iterations, default=20

    Returns:
    --------
    float
        A real root of the polynomial (NaN if not converged)
    """

    def polynomial(p, x):
        return jnp.polyval(p, x)

    def derivative(p, x):
        dp = jnp.polyder(p)
        return jnp.polyval(dp, x)

    def second_derivative(p, x):
        d2p = jnp.polyder(jnp.polyder(p))
        return jnp.polyval(d2p, x)

    def laguerre_step(x, _):
        f_x = polynomial(coeffs, x)
        df_x = derivative(coeffs, x)
        d2f_x = second_derivative(coeffs, x)

        G = df_x / (f_x + 1e-10)  # Avoid division by zero
        H = G ** 2 - d2f_x / (f_x + 1e-10)

        denom1 = G + jnp.sqrt((coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))
        denom2 = G - jnp.sqrt((coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))

        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        x_new = x - (coeffs.shape[0] - 1) / (denom + 1e-10)
        return x_new, jnp.abs(x_new - x) < tol

    # Run Laguerre's method iteratively
    root, converged = jax.lax.scan(laguerre_step, init_guess, None, length=max_iter)

    return jnp.where(converged, root, jnp.nan)  # Return NaN if not converged


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
