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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
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
from ..config.Config import *

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

    # First quadrant
    F_pad_fft = F_pad_fft.at[:nx_rfft, :ny_rfft, :].set(F_fft[:nx_rfft, :ny_rfft, :])

    # Second quadrant
    F_pad_fft = F_pad_fft.at[mx - (nx_rfft-1):, :ny_rfft, :].set(F_fft[(nx_rfft-1):, :ny_rfft, :])

    # Transform back to spatial domain using irfft2
    F_pad = jnp.fft.irfft2(F_pad_fft, axes=(0, 1), s=(mx, my))

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

    # Transform back to physical space and apply 9/4 scaling
    F = (9.0 / 4.0) * jnp.fft.irfft2(F_fft, axes=(0, 1), s=(nx, ny))

    return F
