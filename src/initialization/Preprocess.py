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
File: Preprocess.py
==========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: generates static variables which will be re-used
numerous times during a simulation
"""


# ============================================================
#  Imports
# ============================================================

import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *


# ============================================================
#  FFT-related constants
# ============================================================


def Constant():
    """ Note: I am using _loc in variable names to distinguish
    them from outer scope variable names"""

    # Determine padded dimensions
    mx_loc = int(1.5 * nx)
    my_loc = int(1.5 * ny)

    # Number of Fourier modes in x- and y-directions for input arrays
    nx_rfft_loc = nx // 2 + 1
    ny_rfft_loc = ny // 2 + 1

    # Number of Fourier modes in x- and y-directions for padded arrays
    mx_rfft_loc = mx_loc // 2 + 1
    my_rfft_loc = my_loc // 2 + 1

    return (mx_loc, my_loc,
            nx_rfft_loc, ny_rfft_loc,
            mx_rfft_loc, my_rfft_loc)


mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ============================================================
# Wavenumbers related to real FFT
# ============================================================


def Wavenumber():
    """
    Parameters:
    -----------
    None: Uses global parameters from Config

    Returns:
    --------
    kx2 : ndarray, shape (nx, ny//2 + 1, nz)
        Wavenumber array for x-direction derivatives, broadcast to match FFT dimensions
    ky2 : ndarray, shape (nx, ny//2 + 1, nz)
        Wavenumber array for y-direction derivatives, broadcast to match FFT dimensions

    Note: rfftfreq is used for y as we are using real FFT
    """

    kx = jnp.fft.fftfreq(nx, 1 / nx)
    ky = jnp.fft.rfftfreq(ny, 1 / ny)

    # Zeroing Nyquist frequencies to avoid instabilities
    kx = kx.at[nx // 2].set(0)
    ky = ky.at[ny // 2].set(0)

    kx2 = kx[:, None, None]  # Reshape for broadcasting
    kx2 = jnp.broadcast_to(kx2, (nx, ny_rfft, nz))

    ky2 = ky[None, :, None]  # Reshape for broadcasting
    ky2 = jnp.broadcast_to(ky2, (nx, ny_rfft, nz))

    return kx2, ky2


# ============================================================
# Array Initialization Functions
# ============================================================


def ZeRo3DIni():
    """Create an array of zeros."""
    if use_double_precision:
        ZeRo = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    else:
        ZeRo = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

    return ZeRo


def ZeRo2DIni():
    """Create an array of zeros."""
    if use_double_precision:
        ZeRo2D = jnp.zeros((nx, ny), dtype=jnp.float64)
    else:
        ZeRo2D = jnp.zeros((nx, ny), dtype=jnp.float32)

    return ZeRo2D


def ZeRo1DIni():
    """Create an array of zeros."""
    if use_double_precision:
        ZeRo1D = jnp.zeros((nz), dtype=jnp.float64)
    else:
        ZeRo1D = jnp.zeros((nz), dtype=jnp.float32)

    return ZeRo1D


def ZeRo3D_fftIni():
    """Create an array of zeros for Fourier space operations with rfft2"""
    if use_double_precision:
        ZeRo3D_fft = jnp.zeros((nx, ny_rfft, nz), dtype=jnp.complex128)
    else:
        ZeRo3D_fft = jnp.zeros((nx, ny_rfft, nz), dtype=jnp.complex64)

    return ZeRo3D_fft


def ZeRo3D_padIni():
    """Create a padded array of zeros"""
    if use_double_precision:
        ZeRo3D_pad_fft = jnp.zeros((mx, my, nz), dtype=jnp.float64)
    else:
        ZeRo3D_pad_fft = jnp.zeros((mx, my, nz), dtype=jnp.float32)

    return ZeRo3D_pad_fft


def ZeRo3D_pad_fftIni():
    """Create a padded array of zeros for Fourier space operations with rfft2"""
    if use_double_precision:
        ZeRo3D_pad_fft = jnp.zeros((mx, my_rfft, nz), dtype=jnp.complex128)
    else:
        ZeRo3D_pad_fft = jnp.zeros((mx, my_rfft, nz), dtype=jnp.complex64)

    return ZeRo3D_pad_fft
