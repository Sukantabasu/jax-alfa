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
File: ScalarSGSFluxes_STABSM.py
=================================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-22
:Description: SGS scalar flux computation for the stability-dependent
              Smagorinsky model (STAB-SM, optSgs=5).

              Scalar flux formula (no factor of 2):
                q_i = -Lambda(z) * fh(Ri) * |S| * dTH/dxi

              Lambda_uvp2_3D, fhS_uvp, Lambda_w2_3D, fhS_w are pre-computed
              by SGSStresses_STABSM and passed in to avoid redundant work.
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

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1, Dealias2


# ============================================================
# Scalar fluxes — with dealiasing
# ============================================================

@jax.jit
def ScalarFluxes_Dealias_STABSM(
        Lambda_uvp2_3D,
        Lambda_w2_3D,
        fhS_uvp,
        fhS_w,
        dTHdx, dTHdy, dTHdz,
        qz_sfc,
        ZeRo3D_fft,
        ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    Lambda_uvp2_3D : ndarray (nx, ny, nz)
        3-D Lambda field at UVP nodes (from SGSStresses_STABSM)
    Lambda_w2_3D : ndarray (nx, ny, nz)
        3-D Lambda field at W nodes (from SGSStresses_STABSM)
    fhS_uvp : ndarray (nx, ny, nz)
        fh * |S| at UVP nodes (from SGSStresses_STABSM)
    fhS_w : ndarray (nx, ny, nz)
        fh * |S| at W nodes (from SGSStresses_STABSM)
    dTHdx, dTHdy : ndarray (nx, ny, nz)
        Horizontal potential temperature gradients (on UVP nodes)
    dTHdz : ndarray (nx, ny, nz)
        Vertical potential temperature gradient (on W nodes)
    qz_sfc : ndarray (nx, ny)
        Surface sensible heat flux
    ZeRo3D_fft : ndarray
        Pre-allocated FFT zero array for Dealias2
    ZeRo3D_pad_fft : ndarray
        Pre-allocated padded FFT zero array for Dealias1

    Returns:
    --------
    qx, qy : ndarray (nx, ny, nz)
        Horizontal SGS scalar flux components (at UVP nodes)
    qz : ndarray (nx, ny, nz)
        Vertical SGS scalar flux component (at W nodes)
    """

    Lambda_uvp2_pad = Dealias1(FFT(Lambda_uvp2_3D), ZeRo3D_pad_fft)
    Lambda_w2_pad   = Dealias1(FFT(Lambda_w2_3D),   ZeRo3D_pad_fft)
    fhS_uvp_pad     = Dealias1(FFT(fhS_uvp),        ZeRo3D_pad_fft)
    fhS_w_pad       = Dealias1(FFT(fhS_w),          ZeRo3D_pad_fft)

    dTHdx_pad = Dealias1(FFT(dTHdx), ZeRo3D_pad_fft)
    dTHdy_pad = Dealias1(FFT(dTHdy), ZeRo3D_pad_fft)
    dTHdz_pad = Dealias1(FFT(dTHdz), ZeRo3D_pad_fft)

    # Horizontal fluxes at UVP nodes (no factor of 2 — STAB-SM convention)
    qx_pad = -Lambda_uvp2_pad * fhS_uvp_pad * dTHdx_pad
    qy_pad = -Lambda_uvp2_pad * fhS_uvp_pad * dTHdy_pad

    qx_pad = qx_pad.at[:, :, nz - 1].set(0)
    qy_pad = qy_pad.at[:, :, nz - 1].set(0)

    qx = Dealias2(FFT_pad(qx_pad), ZeRo3D_fft)
    qy = Dealias2(FFT_pad(qy_pad), ZeRo3D_fft)

    # Vertical flux at W nodes — Lambda_w is directly on W nodes (no StagGridAvg)
    qz_pad = jnp.zeros_like(fhS_w_pad)
    qz_pad = qz_pad.at[:, :, 1:nz - 1].set(
        -Lambda_w2_pad[:, :, 1:nz - 1] *
        fhS_w_pad[:, :, 1:nz - 1] *
        dTHdz_pad[:, :, 1:nz - 1])

    qz_pad = qz_pad.at[:, :, nz - 1].set(0)

    qz = Dealias2(FFT_pad(qz_pad), ZeRo3D_fft)
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qx, qy, qz


# ============================================================
# Scalar fluxes — without dealiasing
# ============================================================

@jax.jit
def ScalarFluxes_NoDealias_STABSM(
        Lambda_uvp2_3D,
        Lambda_w2_3D,
        fhS_uvp,
        fhS_w,
        dTHdx, dTHdy, dTHdz,
        qz_sfc):
    """
    Parameters:
    -----------
    Lambda_uvp2_3D : ndarray (nx, ny, nz)
        3-D Lambda field at UVP nodes
    Lambda_w2_3D : ndarray (nx, ny, nz)
        3-D Lambda field at W nodes
    fhS_uvp : ndarray (nx, ny, nz)
        fh * |S| at UVP nodes
    fhS_w : ndarray (nx, ny, nz)
        fh * |S| at W nodes
    dTHdx, dTHdy : ndarray (nx, ny, nz)
        Horizontal potential temperature gradients
    dTHdz : ndarray (nx, ny, nz)
        Vertical potential temperature gradient
    qz_sfc : ndarray (nx, ny)
        Surface sensible heat flux

    Returns:
    --------
    qx, qy : ndarray (nx, ny, nz)
        Horizontal SGS scalar flux components
    qz : ndarray (nx, ny, nz)
        Vertical SGS scalar flux component
    """

    qx = -Lambda_uvp2_3D * fhS_uvp * dTHdx
    qy = -Lambda_uvp2_3D * fhS_uvp * dTHdy

    qx = qx.at[:, :, nz - 1].set(0)
    qy = qy.at[:, :, nz - 1].set(0)

    qz = jnp.zeros_like(fhS_w)
    qz = qz.at[:, :, 1:nz - 1].set(
        -Lambda_w2_3D[:, :, 1:nz - 1] *
        fhS_w[:, :, 1:nz - 1] *
        dTHdz[:, :, 1:nz - 1])

    qz = qz.at[:, :, nz - 1].set(0)
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qx, qy, qz
