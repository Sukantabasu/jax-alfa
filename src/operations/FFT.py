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
File: FFT.py
==================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: performs FFT-based operations using JAX
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

from ..initialization.Preprocess import Constant
mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ============================================================
#  Compute real fft of a variable F
# ============================================================


@jax.jit
def FFT(F):
    """
    Parameters:
    -----------
    F : ndarray
        3D input field with shape (nx, ny, nz)

    Returns:
    --------
    F_fft : ndarray in Jax format
        rfft2 of a field
    """

    F_fft = jnp.fft.rfft2(F, axes=(0, 1))

    return F_fft


# ============================================================
#  Compute real fft of a padded variable F_pad
# ============================================================


@jax.jit
def FFT_pad(F_pad):
    """
    Parameters:
    -----------
    F_pad : ndarray with shape (mx=1.5*nx, my=1.5*ny, nz)
        3D input field

    Returns:
    --------
    F_pad_fft : ndarray in Jax format
        rfft2 of a padded field
    """

    # Compute 2D real FFT along x and y axes
    F_pad_fft = jnp.fft.rfft2(F_pad, axes=(0, 1))

    return F_pad_fft
