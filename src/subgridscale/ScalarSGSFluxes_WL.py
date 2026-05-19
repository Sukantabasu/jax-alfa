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
File: ScalarSGSFluxes_WL.py
============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-9
:Description: computes SGS scalar fluxes using the Wong-Lilly (1994)
              SGS base model.  Flux formula:
                q_i = -(C_WL/Pr_t) * Delta^(4/3) * (dTH/dx_i)
              (no strain-rate magnitude factor; filter width exponent
              4/3 instead of 2; no leading factor of 2).
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import helper functions
from ..utilities.Utilities import StagGridAvg

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1, Dealias2


# ======================================================
# Compute SGS scalar fluxes on UVP nodes with dealiasing
# ======================================================

@jax.jit
def ScalarFluxesUVPnodes_Dealias(
        dTHdx_pad, dTHdy_pad,
        CwlPrRatio_3D_pad,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdx_pad, dTHdy_pad : ndarray
        Dealiased potential temperature gradients
    CwlPrRatio_3D_pad : ndarray
        Dealiased C_WL/Pr_t field
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components in x and y directions
    """

    preCompute = -(L ** (4 / 3)) * CwlPrRatio_3D_pad
    qx_pad = preCompute * dTHdx_pad
    qy_pad = preCompute * dTHdy_pad

    qx_pad = qx_pad.at[:, :, nz - 1].set(0)
    qy_pad = qy_pad.at[:, :, nz - 1].set(0)

    qx = Dealias2(FFT_pad(qx_pad), ZeRo3D_fft)
    qy = Dealias2(FFT_pad(qy_pad), ZeRo3D_fft)

    return qx, qy


# =========================================================
# Compute SGS scalar fluxes on UVP nodes without dealiasing
# =========================================================

@jax.jit
def ScalarFluxesUVPnodes_NoDealias(
        dTHdx, dTHdy,
        CwlPrRatio_3D):
    """
    Parameters:
    -----------
    dTHdx, dTHdy : ndarray
        Potential temperature gradients at UVP nodes
    CwlPrRatio_3D : ndarray
        C_WL/Pr_t field

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components in x and y directions
    """

    preCompute = -(L ** (4 / 3)) * CwlPrRatio_3D
    qx = preCompute * dTHdx
    qy = preCompute * dTHdy

    qx = qx.at[:, :, nz - 1].set(0)
    qy = qy.at[:, :, nz - 1].set(0)

    return qx, qy


# ====================================================
# Compute SGS scalar fluxes on W nodes with dealiasing
# ====================================================

@jax.jit
def ScalarFluxesWnodes_Dealias(
        dTHdz_pad,
        CwlPrRatio_3D_pad,
        qz_sfc,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdz_pad : ndarray
        Dealiased potential temperature gradient in z-direction
    CwlPrRatio_3D_pad : ndarray
        Dealiased C_WL/Pr_t field
    qz_sfc : ndarray
        Surface heat flux
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction
    """

    qz_pad = jnp.zeros_like(dTHdz_pad)

    qz_pad = qz_pad.at[:, :, 1:nz - 1].set(
        -(L ** (4 / 3)) *
        StagGridAvg(CwlPrRatio_3D_pad[:, :, :nz - 1]) *
        dTHdz_pad[:, :, 1:nz - 1]
    )

    qz_pad = qz_pad.at[:, :, nz - 1].set(0)

    qz = Dealias2(FFT_pad(qz_pad), ZeRo3D_fft)
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qz


# =======================================================
# Compute SGS scalar fluxes on W nodes without dealiasing
# =======================================================

@jax.jit
def ScalarFluxesWnodes_NoDealias(
        dTHdz,
        CwlPrRatio_3D,
        qz_sfc):
    """
    Parameters:
    -----------
    dTHdz : ndarray
        Potential temperature gradient in z-direction
    CwlPrRatio_3D : ndarray
        C_WL/Pr_t field
    qz_sfc : ndarray
        Surface heat flux

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction
    """

    qz = jnp.zeros_like(dTHdz)

    qz = qz.at[:, :, 1:nz - 1].set(
        -(L ** (4 / 3)) *
        StagGridAvg(CwlPrRatio_3D[:, :, :nz - 1]) *
        dTHdz[:, :, 1:nz - 1]
    )

    qz = qz.at[:, :, nz - 1].set(0)
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qz
