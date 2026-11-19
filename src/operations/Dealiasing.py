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
File: Dealiasing.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: performs dealiasing
"""


# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *

# Import constants
from ..initialization.Preprocess import Constant
mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ============================================================
# First function for dealiasing
# ============================================================

@jax.jit
def Dealias1(F_fft, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    F_fft : jnp.ndarray
        Fourier transformed input array
    ZeRo3D_pad_fft : jnp.ndarray
        Pre-allocated zero-padded array

    Returns:
    --------
    F_pad : jnp.ndarray
        Dealiased padded array in spatial domain
    """

    # Allocate padded array
    F_pad_fft = ZeRo3D_pad_fft.copy()

    # Copy all modes except the x- and y-Nyquist modes, which are not Nyquist
    # modes on the padded grid and would otherwise be doubled (they are also
    # zeroed in Dealias2 and in the spectral derivatives). Corrected 2026-11-19.

    # First quadrant (kx >= 0)
    F_pad_fft = F_pad_fft.at[:(nx_rfft-1), :(ny_rfft-1), :].set(F_fft[:(nx_rfft-1), :(ny_rfft-1), :])

    # Second quadrant (kx < 0)
    F_pad_fft = F_pad_fft.at[mx - (nx_rfft-2):, :(ny_rfft-1), :].set(F_fft[nx_rfft:, :(ny_rfft-1), :])

    # Transform back to spatial domain using irfft2
    # Scale by (mx*my)/(nx*ny) = 9/4 so that the padded field has the same
    # physical amplitude as the input. With this convention Dealias2 applies
    # 4/9 and products of ANY number of padded fields are normalized
    # correctly (before 2026-11-19 the padded fields carried a factor 4/9 and
    # Dealias2 applied 9/4, which is exact only for products of two fields;
    # the three-factor SGS products were attenuated by 4/9 when FGR = 1).
    F_pad = (9.0 / 4.0) * jnp.fft.irfft2(F_pad_fft, axes=(0, 1), s=(mx, my))

    return F_pad


# ============================================================
# Second function for dealiasing
# ============================================================

@jax.jit
def Dealias2(F_pad_fft, ZeRo3D_fft):
    """
    Parameters:
    -----------
    F_pad_fft : jnp.ndarray
        Fourier transformed padded array
    ZeRo3D_fft : jnp.ndarray
        Pre-allocated zero array for Fourier operations

    Returns:
    --------
    F : jnp.ndarray
        Dealiased output array on regular grid
    Note: Nyquist is explicitly set to zero.
    """

    # Allocate array
    F_fft = ZeRo3D_fft.copy()

    # First quadrant
    F_fft = F_fft.at[:(nx_rfft-1), :(ny_rfft-1), :].set(F_pad_fft[:(nx_rfft-1), :(ny_rfft-1), :])

    # Second quadrant
    F_fft = F_fft.at[nx_rfft:, :(ny_rfft-1), :].set(F_pad_fft[mx - (nx_rfft-2):, :(ny_rfft-1), :])

    # Transform back to physical space; the factor 4/9 = (nx*ny)/(mx*my)
    # undoes the change of FFT normalization between the two grid sizes
    F = (4.0 / 9.0) * jnp.fft.irfft2(F_fft, axes=(0, 1), s=(nx, ny))

    return F
