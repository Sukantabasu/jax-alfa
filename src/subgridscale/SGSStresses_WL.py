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
File: SGSStresses_WL.py
=======================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-9
:Description: computes SGS stresses using the Wong-Lilly (1994) SGS
              base model.  Stress formula:
                tau_ij = -2 * C_WL * Delta^(4/3) * S_ij
              (no strain-rate magnitude factor; filter width exponent
              4/3 instead of 2).
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

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1, Dealias2

# Import strain rates functions
from .StrainRates import StrainsUVPnodes_Dealias, StrainsWnodes_Dealias
from .StrainRates import StrainsUVPnodes_NoDealias, StrainsWnodes_NoDealias

# Import LASDD-WL
from .DynamicSGS_LASDD_WL import LASDD
from .DynamicSGS_ScalarLASDD_WL import ScalarLASDD

# Import helper functions
from ..utilities.Utilities import StagGridAvg

# Import wall model from SM file (model-agnostic surface BC)
from .SGSStresses_SM import Wall


# =================================================
# Compute SGS stresses on UVP nodes with dealiasing
# =================================================

@jax.jit
def StressesUVPnodes_Dealias(
        S11_pad, S22_pad, S33_pad, S12_pad,
        Cwl_3D_pad,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    S11_pad, S22_pad, S33_pad, S12_pad : ndarray
        Dealiased strain rate components
    Cwl_3D_pad : ndarray
        Dealiased Wong-Lilly coefficient C_WL field
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    """

    preCompute = -2 * (L ** (4 / 3)) * Cwl_3D_pad
    txx_pad = preCompute * S11_pad
    tyy_pad = preCompute * S22_pad
    tzz_pad = preCompute * S33_pad
    txy_pad = preCompute * S12_pad

    txx_pad = txx_pad.at[:, :, nz - 1].set(0)
    tyy_pad = tyy_pad.at[:, :, nz - 1].set(0)
    tzz_pad = tzz_pad.at[:, :, nz - 1].set(0)
    txy_pad = txy_pad.at[:, :, nz - 1].set(0)

    txx = Dealias2(FFT_pad(txx_pad), ZeRo3D_fft)
    tyy = Dealias2(FFT_pad(tyy_pad), ZeRo3D_fft)
    tzz = Dealias2(FFT_pad(tzz_pad), ZeRo3D_fft)
    txy = Dealias2(FFT_pad(txy_pad), ZeRo3D_fft)

    return txx, tyy, tzz, txy


# ====================================================
# Compute SGS stresses on UVP nodes without dealiasing
# ====================================================

@jax.jit
def StressesUVPnodes_NoDealias(
        S11, S22, S33, S12,
        Cwl_3D):
    """
    Parameters:
    -----------
    S11, S22, S33, S12 : ndarray
        Strain rate components
    Cwl_3D : ndarray
        Wong-Lilly coefficient C_WL field

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    """

    preCompute = -2 * (L ** (4 / 3)) * Cwl_3D
    txx = preCompute * S11
    tyy = preCompute * S22
    tzz = preCompute * S33
    txy = preCompute * S12

    txx = txx.at[:, :, nz - 1].set(0)
    tyy = tyy.at[:, :, nz - 1].set(0)
    tzz = tzz.at[:, :, nz - 1].set(0)
    txy = txy.at[:, :, nz - 1].set(0)

    return txx, tyy, tzz, txy


# ===============================================
# Compute SGS stresses on W nodes with dealiasing
# ===============================================

@jax.jit
def StressesWnodes_Dealias(
        S13_pad, S23_pad,
        Cwl_3D_pad,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    S13_pad, S23_pad : ndarray
        Dealiased strain rate components
    Cwl_3D_pad : ndarray
        Dealiased Wong-Lilly coefficient C_WL field
    u, v : ndarray
        Velocity components for wall model
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    """

    txz_pad = jnp.zeros_like(S13_pad)
    tyz_pad = jnp.zeros_like(S13_pad)

    preCompute = (-2 * (L ** (4 / 3)) *
                  StagGridAvg(Cwl_3D_pad[:, :, :nz - 1]))
    txz_pad = txz_pad.at[:, :, 1:nz - 1].set(preCompute *
                                              S13_pad[:, :, 1:nz - 1])
    tyz_pad = tyz_pad.at[:, :, 1:nz - 1].set(preCompute *
                                              S23_pad[:, :, 1:nz - 1])

    txz_pad = txz_pad.at[:, :, nz - 1].set(0)
    tyz_pad = tyz_pad.at[:, :, nz - 1].set(0)

    txz = Dealias2(FFT_pad(txz_pad), ZeRo3D_fft)
    tyz = Dealias2(FFT_pad(tyz_pad), ZeRo3D_fft)

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
        Cwl_3D,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0):
    """
    Parameters:
    -----------
    S13, S23 : ndarray
        Strain rate components
    Cwl_3D : ndarray
        Wong-Lilly coefficient C_WL field
    u, v : ndarray
        Velocity components for wall model
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    """

    txz = jnp.zeros_like(S13)
    tyz = jnp.zeros_like(S13)

    preCompute = (-2 * (L ** (4 / 3)) *
                  StagGridAvg(Cwl_3D[:, :, :nz - 1]))
    txz = txz.at[:, :, 1:nz - 1].set(preCompute * S13[:, :, 1:nz - 1])
    tyz = tyz.at[:, :, 1:nz - 1].set(preCompute * S23[:, :, 1:nz - 1])

    txz = txz.at[:, :, nz - 1].set(0)
    tyz = tyz.at[:, :, nz - 1].set(0)

    txz_wall, tyz_wall = Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0)
    txz = txz.at[:, :, 0].set(txz_wall)
    tyz = tyz.at[:, :, 0].set(tyz_wall)

    return txz, tyz