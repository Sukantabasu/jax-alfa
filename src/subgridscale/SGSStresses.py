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
File: SGSStresses.py
========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-7
:Description: computes SGS stresses
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

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1, Dealias2

# Import strain rates functions
from .StrainRates import StrainsUVPnodes_Dealias, StrainsWnodes_Dealias
from .StrainRates import StrainsUVPnodes_NoDealias, StrainsWnodes_NoDealias

# Import LASDD
from .DynamicSGS_LASDD import LASDD
from .DynamicSGS_ScalarLASDD import ScalarLASDD

# Import helper functions
from ..utilities.Utilities import StagGridAvg


# ============================================================
# Wall function to compute surface stresses
# ============================================================

@jax.jit
def Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0):
    """
    Computes surface stresses using Monin-Obukhov similarity theory.

    Parameters:
    -----------
    u, v : ndarray
        Velocity components near the surface
    M_sfc_loc : ndarray
        Near-surface wind speed
    psi2D_m, psi2D_m0 : ndarray
        Stability correction functions at first level and surface

    Returns:
    --------
    txz, tyz : ndarray
        Surface stresses in x and y directions
    """

    # Compute denominator for friction velocity
    denom = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0

    # Compute friction velocity
    ustar = vonk * M_sfc_loc / denom

    # Compute wall stresses
    txz = -(ustar ** 2) * (u[:, :, 0] + Ugal) / M_sfc_loc
    tyz = -(ustar ** 2) * v[:, :, 0] / M_sfc_loc

    return txz, tyz


# ============================================================
# Compute SGS stresses on UVP nodes with dealiasing
# ============================================================

@jax.jit
def StressesUVPnodes_Dealias(
        S11_pad, S22_pad, S33_pad, S12_pad,
        S_uvp_pad,
        Cs2_3D_pad,
        ZeRo3D_fft):
    """
    Computes SGS stresses at UVP nodes with dealiasing.

    Parameters:
    -----------
    S11_pad, S22_pad, S33_pad, S12_pad : ndarray
        Dealiased strain rate components
    S_uvp_pad : ndarray
        Dealiased strain rate magnitude
    Cs2_3D_pad : ndarray
        Dealiased Smagorinsky coefficient field
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for FFT operations

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    """

    # Compute SGS stresses on uvp nodes
    preCompute = -2 * (L ** 2) * Cs2_3D_pad * S_uvp_pad
    txx_pad = preCompute * S11_pad
    tyy_pad = preCompute * S22_pad
    tzz_pad = preCompute * S33_pad
    txy_pad = preCompute * S12_pad

    # Set top boundary conditions
    txx_pad = txx_pad.at[:, :, nz - 1].set(0)
    tyy_pad = tyy_pad.at[:, :, nz - 1].set(0)
    tzz_pad = tzz_pad.at[:, :, nz - 1].set(0)
    txy_pad = txy_pad.at[:, :, nz - 1].set(0)

    # Apply dealiasing
    txx = Dealias2(FFT_pad(txx_pad), ZeRo3D_fft)
    tyy = Dealias2(FFT_pad(tyy_pad), ZeRo3D_fft)
    tzz = Dealias2(FFT_pad(tzz_pad), ZeRo3D_fft)
    txy = Dealias2(FFT_pad(txy_pad), ZeRo3D_fft)

    return txx, tyy, tzz, txy


# ============================================================
# Compute SGS stresses on UVP nodes without dealiasing
# ============================================================


@jax.jit
def StressesUVPnodes_NoDealias(
        S11, S22, S33, S12,
        S_uvp,
        Cs2_3D):
    """
    Computes SGS stresses at UVP nodes without dealiasing.

    Parameters:
    -----------
    S11, S22, S33, S12 : ndarray
        Strain rate components
    S_uvp : ndarray
        Strain rate magnitude
    Cs2_3D : ndarray
        Smagorinsky coefficient field

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    """

    # Compute SGS stresses on uvp nodes
    preCompute = -2 * (L ** 2) * Cs2_3D * S_uvp
    txx = preCompute * S11
    tyy = preCompute * S22
    tzz = preCompute * S33
    txy = preCompute * S12

    # Set top boundary conditions
    txx = txx.at[:, :, nz - 1].set(0)
    tyy = tyy.at[:, :, nz - 1].set(0)
    tzz = tzz.at[:, :, nz - 1].set(0)
    txy = txy.at[:, :, nz - 1].set(0)

    return txx, tyy, tzz, txy


# ============================================================
# Compute SGS stresses on W nodes with dealiasing
# ============================================================

@jax.jit
def StressesWnodes_Dealias(
        S13_pad, S23_pad,
        S_w_pad,
        Cs2_3D_pad,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D_fft):
    """
    Computes SGS stresses at W nodes with dealiasing.

    Parameters:
    -----------
    S13_pad, S23_pad : ndarray
        Dealiased strain rate components
    S_w_pad : ndarray
        Dealiased strain rate magnitude at W nodes
    Cs2_3D_pad : ndarray
        Dealiased Smagorinsky coefficient field
    u, v : ndarray
        Velocity components for wall model
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for FFT operations

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    """

    # Initialize arrays
    txz_pad = jnp.zeros_like(S_w_pad)
    tyz_pad = jnp.zeros_like(S_w_pad)

    # Interior points
    txz_pad = txz_pad.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2_3D_pad[:, :, :nz - 1]) *
        S_w_pad[:, :, 1:nz - 1] * S13_pad[:, :, 1:nz - 1]
    )
    tyz_pad = tyz_pad.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2_3D_pad[:, :, :nz - 1]) *
        S_w_pad[:, :, 1:nz - 1] * S23_pad[:, :, 1:nz - 1]
    )

    # Top boundary conditions
    txz_pad = txz_pad.at[:, :, nz - 1].set(0)
    tyz_pad = tyz_pad.at[:, :, nz - 1].set(0)

    # Apply dealiasing
    txz = Dealias2(FFT_pad(txz_pad), ZeRo3D_fft)
    tyz = Dealias2(FFT_pad(tyz_pad), ZeRo3D_fft)

    # Apply wall model for bottom boundary
    txz_wall, tyz_wall = Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0)
    txz = txz.at[:, :, 0].set(txz_wall)
    tyz = tyz.at[:, :, 0].set(tyz_wall)

    return txz, tyz


# ============================================================
# Compute SGS stresses on W nodes without dealiasing
# ============================================================

@jax.jit
def StressesWnodes_NoDealias(
        S13, S23,
        S_w,
        Cs2_3D,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0):
    """
    Computes SGS stresses at W nodes without dealiasing.

    Parameters:
    -----------
    S13, S23 : ndarray
        Strain rate components
    S_w : ndarray
        Strain rate magnitude at W nodes
    Cs2_3D : ndarray
        Smagorinsky coefficient field
    u, v : ndarray
        Velocity components for wall model
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    """

    # Initialize arrays
    txz = jnp.zeros_like(S_w)
    tyz = jnp.zeros_like(S_w)

    # Interior points
    txz = txz.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2_3D[:, :, :nz - 1]) *
        S_w[:, :, 1:nz - 1] * S13[:, :, 1:nz - 1]
    )
    tyz = tyz.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2_3D[:, :, :nz - 1]) *
        S_w[:, :, 1:nz - 1] * S23[:, :, 1:nz - 1]
    )

    # Top boundary conditions
    txz = txz.at[:, :, nz - 1].set(0)
    tyz = tyz.at[:, :, nz - 1].set(0)

    # Apply wall model for bottom boundary
    txz_wall, tyz_wall = Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0)
    txz = txz.at[:, :, 0].set(txz_wall)
    tyz = tyz.at[:, :, 0].set(tyz_wall)

    return txz, tyz
