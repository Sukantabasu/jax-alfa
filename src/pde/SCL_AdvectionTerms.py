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
File: SCL_AdvectionTerms.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: computes the advective terms of scalar transport
"""


# ============================================
# Imports
# ============================================

import jax
from ..config.Config import *
from ..config import Config

# Import derived variables
from ..config.DerivedVars import *

from ..utilities.Utilities import StagGridAvg

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealias functions
from ..operations.Dealiasing import Dealias1, Dealias2

# Import constants
from ..initialization.Preprocess import Constant
mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# -----------------------------------------------------------------------------
# Use dealiasing in computing the advection terms of scalar transport
# -----------------------------------------------------------------------------

@jax.jit
def ScalarAdvection_Dealias(u, v, w,
                            dsdx, dsdy, dsdz,
                            ZeRo3D_fft, ZeRo3D_pad, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components in x, y, and z directions respectively
    dsdx, dsdy, dsdz : ndarray of shape (nx, ny, nz)
        Derivatives of a scalar with respect to x, y and z
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for Fourier operations
    ZeRo3D_pad : ndarray
        Pre-allocated zero-padded array
    ZeRo3D_pad_fft : ndarray
        Pre-allocated zero-padded array for Fourier operations

    Returns:
    --------
    scalarAdvectionSum : ndarray of shape (nx, ny, nz)
        Sum of convective terms with dealiasing applied
    """

    # Dealias - forward operation
    u_pad = Dealias1(FFT(u), ZeRo3D_pad_fft)
    v_pad = Dealias1(FFT(v), ZeRo3D_pad_fft)
    w_pad = Dealias1(FFT(w), ZeRo3D_pad_fft)
    dsdx_pad = Dealias1(FFT(dsdx), ZeRo3D_pad_fft)
    dsdy_pad = Dealias1(FFT(dsdy), ZeRo3D_pad_fft)
    dsdz_pad = Dealias1(FFT(dsdz), ZeRo3D_pad_fft)

    # Compute advection terms
    cc1 = u_pad * dsdx_pad + v_pad * dsdy_pad
    cc2 = w_pad * dsdz_pad

    # Initialize array for combined convective terms
    cc = ZeRo3D_pad.copy()

    # Compute convective terms with grid averaging for interior points
    cc = cc.at[:, :, 0:nz - 1].set(cc1[:, :, 0:nz - 1] +
                                   StagGridAvg(cc2))

    # Handle top boundary
    cc = cc.at[:, :, nz - 1].set(cc1[:, :, nz - 1] +
                                 cc2[:, :, nz - 1])

    # Dealias the convective terms - inverse operation
    scalarAdvectionSum = Dealias2(FFT_pad(cc), ZeRo3D_fft)

    return scalarAdvectionSum


# ------------------------------------------------------------------------------
# Compute the advection terms of scalar transport without any dealiasing
# ------------------------------------------------------------------------------

@jax.jit
def ScalarAdvection_NoDealias(u, v, w,
                              dsdx, dsdy, dsdz,
                              ZeRo3D):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components in x, y, and z directions respectively
    dsdx, dsdy, dsdz : ndarray of shape (nx, ny, nz)
        Derivatives of a scalar with respect to x, y and z
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for calculations

    Returns:
    --------
    scalarAdvectionSum : ndarray of shape (nx, ny, nz)
        Sum of convective terms without dealiasing
    """

    # Compute advection terms
    cc1 = u * dsdx + v * dsdy
    cc2 = w * dsdz

    # Initialize array for combined convective terms
    scalarAdvectionSum = ZeRo3D.copy()

    # Compute convective terms with grid averaging for interior points
    scalarAdvectionSum = (
        scalarAdvectionSum.at[:, :, 0:nz - 1].set(cc1[:, :, 0:nz - 1]
                                                  + StagGridAvg(cc2)))

    # Handle top boundary
    scalarAdvectionSum = (
        scalarAdvectionSum.at[:, :, nz - 1].set(cc1[:, :, nz - 1]
                                                + cc2[:, :, nz - 1]))

    return scalarAdvectionSum


# ------------------------------------------------------------------------------
# Select one of the above functions based on dealias flag
# ------------------------------------------------------------------------------

@jax.jit
def ScalarAdvection(
        u, v, w,
        dsdx, dsdy, dsdz,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad,
        ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components in x, y, and z directions respectively
    dsdx, dsdy, dsdz : ndarray of shape (nx, ny, nz)
        Derivatives of a scalar with respect to x, y and z
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for calculations
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for Fourier operations
    ZeRo3D_pad : ndarray
        Pre-allocated zero-padded array
    ZeRo3D_pad_fft : ndarray
        Pre-allocated zero-padded array for Fourier operations

    Returns:
    --------
    scalarAdvectionSum : ndarray of shape (nx, ny, nz)
        Sum of advection terms from either dealiased or non-dealiased method
    """

    if optDealias == 1:

        scalarAdvectionSum = (
            ScalarAdvection_Dealias(
                u, v, w,
                dsdx, dsdy, dsdz,
                ZeRo3D_fft, ZeRo3D_pad, ZeRo3D_pad_fft))

    else:

        scalarAdvectionSum = (
            ScalarAdvection_NoDealias(
                u, v, w,
                dsdx, dsdy, dsdz,
                ZeRo3D))

    return scalarAdvectionSum
