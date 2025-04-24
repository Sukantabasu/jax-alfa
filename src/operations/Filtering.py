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
File: Filtering.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: performs main filtering operations for LES
"""


# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *

# Import constants and arrays
from ..initialization.Preprocess import ZeRo3D_fftIni


# ============================================================
# Explicit filtering (for FGR = 1, remove Nyquist)
# ============================================================

@jax.jit
def Filtering_Explicit(F_fft):
    """
    Parameters:
    -----------
    F_fft : ndarray with shape (nx, ny/2, nz)
        rfft2 of a field (could be velocity components or scalars)

    Returns:
    --------
    F_new : ndarray with shape (nx, ny, nz)
        Filtered field
    F_fft_new : ndarray with shape (nx, ny_rfft, nz)
        Filtered field in Fourier space

    Notes:
    ------
    - For FGR = 1 (implicit filtering), Nyquist frequencies are removed
    """

    # Calculate inverse of 2*FGR for cutoff wavenumber
    Inv_2FGR = 1.0 / (2.0 * FGR)

    # Calculate cutoff indices for real FFT
    nx_cut = round(nx * Inv_2FGR)
    ny_cut = round(ny * Inv_2FGR)

    # Initialize zero array for filtered spectrum
    F_fft_new = ZeRo3D_fftIni()

    # Apply spectral cutoff filter
    # First quadrant
    F_fft_new = F_fft_new.at[:nx_cut, :ny_cut, :].set(F_fft[:nx_cut, :ny_cut, :])

    # Second quadrant
    F_fft_new = F_fft_new.at[nx - nx_cut + 1:, :ny_cut, :].set(F_fft[nx - nx_cut + 1:, :ny_cut, :])

    # Transform back to physical space
    F_new = jnp.fft.irfft2(F_fft_new, axes=(0, 1), s=(nx, ny))

    return F_new, F_fft_new


# ============================================================
# Level 1 filtering (filter width = FGR*TFR)
# ============================================================

@jax.jit
def Filtering_Level1(F_fft):
    """
    Parameters:
    -----------
    F_fft : ndarray with shape (nx, ny/2, nz)
        rfft2 of a field (could be velocity components or scalars)

    Returns:
    --------
    F_hat : ndarray with shape (nx, ny, nz)
        Filtered field
    """

    # Calculate cutoff wavenumbers for filtering
    mr = round(nx / (2 * FGR * TFR))  # Cutoff in x-direction for rfft
    mc = round(ny / (2 * FGR * TFR))  # Cutoff in y-direction for rfft

    # Initialize zero array for filtered spectrum
    F_fft_hat = ZeRo3D_fftIni()

    # Apply spectral cutoff filter
    # First quadrant
    F_fft_hat = F_fft_hat.at[:mr, :mc, :].set(F_fft[:mr, :mc, :])

    # Second quadrant
    F_fft_hat = F_fft_hat.at[nx - mr + 1:, :mc, :].set(F_fft[nx - mr + 1:, :mc, :])

    # Transform back to physical space
    F_hat = jnp.fft.irfft2(F_fft_hat, axes=(0, 1), s=(nx, ny))

    return F_hat


# ============================================================
# Level 2 filtering (filter width = FGR*TFR*TFR)
# ============================================================

@jax.jit
def Filtering_Level2(F_fft):
    """
    Parameters:
    -----------
    F_fft : ndarray with shape (nx, ny/2, nz)
        rfft2 of a field (could be velocity components or scalars)

    Returns:
    --------
    F_hatd : ndarray with shape (nx, ny, nz)
        Filtered field with TFR**2 filter width
    """

    # Calculate cutoff wavenumbers for filtering
    pr = round(nx / (2 * FGR * TFR * TFR))  # Cutoff in x-direction for rfft
    pc = round(ny / (2 * FGR * TFR * TFR))  # Cutoff in y-direction for rfft

    # Initialize zero array for filtered spectrum
    F_fft_hatd = ZeRo3D_fftIni()

    # Apply spectral cutoff filter
    # First quadrant
    F_fft_hatd = F_fft_hatd.at[:pr, :pc, :].set(F_fft[:pr, :pc, :])

    # Second quadrant
    F_fft_hatd = F_fft_hatd.at[nx - pr + 1:, :pc, :].set(F_fft[nx - pr + 1:, :pc, :])

    # Transform back to physical space
    F_hatd = jnp.fft.irfft2(F_fft_hatd, axes=(0, 1), s=(nx, ny))

    return F_hatd
