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
File: ScalarSGSFluxes.py
========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: computes SGS scalar fluxes for the eddy-diffusivity model:
              q_i = -2(L^2) * Cs^2/Pr_t * |S| * (∂TH/∂x_i)
              where Cs^2/Pr_t is the model coefficient, |S| is the strain rate
              magnitude, and ∂TH/∂x_i is the potential temperature gradient
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
        S_pad, Cs2PrRatio_3D_pad,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdx_pad, dTHdy_pad : ndarray
        Dealiased potential temperature gradients
    S_pad : ndarray
        Dealiased strain rate magnitude
    Cs2PrRatio_3D_pad : ndarray
        Dealiased Cs^2/Pr_t field
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components in x and y directions
    """

    # Compute SGS scalar fluxes at UVP nodes
    preCompute = -2 * (L ** 2) * Cs2PrRatio_3D_pad * S_pad
    qx_pad = preCompute * dTHdx_pad
    qy_pad = preCompute * dTHdy_pad

    # Set top boundary conditions
    qx_pad = qx_pad.at[:, :, nz - 1].set(0)
    qy_pad = qy_pad.at[:, :, nz - 1].set(0)

    # Apply dealiasing to horizontal fluxes
    qx = Dealias2(FFT_pad(qx_pad), ZeRo3D_fft)
    qy = Dealias2(FFT_pad(qy_pad), ZeRo3D_fft)

    return qx, qy


# =========================================================
# Compute SGS scalar fluxes on UVP nodes without dealiasing
# =========================================================

@jax.jit
def ScalarFluxesUVPnodes_NoDealias(
        dTHdx, dTHdy,
        S, Cs2PrRatio_3D):
    """
    Parameters:
    -----------
    dTHdx, dTHdy : ndarray
        Potential temperature gradients at UVP nodes
    S : ndarray
        Strain rate magnitude at UVP nodes
    Cs2PrRatio_3D : ndarray
        Cs^2/Pr_t field

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components in x and y directions
    """

    # Compute SGS scalar fluxes at UVP nodes
    preCompute = -2 * (L ** 2) * Cs2PrRatio_3D * S
    qx = preCompute * dTHdx
    qy = preCompute * dTHdy

    # Set top boundary conditions
    qx = qx.at[:, :, nz - 1].set(0)
    qy = qy.at[:, :, nz - 1].set(0)

    return qx, qy


# ====================================================
# Compute SGS scalar fluxes on W nodes with dealiasing
# ====================================================

@jax.jit
def ScalarFluxesWnodes_Dealias(
        dTHdz_pad,
        S_pad, Cs2PrRatio_3D_pad,
        qz_sfc,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdz_pad : ndarray
        Dealiased potential temperature gradient in z-direction
    S_pad : ndarray
        Dealiased strain rate magnitude
    Cs2PrRatio_3D_pad : ndarray
        Dealiased Cs^2/Pr_t field
    qz_sfc : ndarray
        Surface heat flux
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for dealiasing

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction
    """

    # Initialize array for vertical flux
    qz_pad = jnp.zeros_like(S_pad)

    # Interior points for vertical flux (on w-nodes)
    qz_pad = qz_pad.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2PrRatio_3D_pad[:, :, :nz - 1]) *
        S_pad[:, :, 1:nz - 1] * dTHdz_pad[:, :, 1:nz - 1]
    )

    # Top boundary condition
    qz_pad = qz_pad.at[:, :, nz - 1].set(0)

    # Apply dealiasing to vertical flux
    qz = Dealias2(FFT_pad(qz_pad), ZeRo3D_fft)

    # Bottom boundary condition
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qz


# =======================================================
# Compute SGS scalar fluxes on W nodes without dealiasing
# =======================================================

@jax.jit
def ScalarFluxesWnodes_NoDealias(
        dTHdz,
        S, Cs2PrRatio_3D,
        qz_sfc):
    """
    Parameters:
    -----------
    dTHdz : ndarray
        Potential temperature gradient in z-direction
    S : ndarray
        Strain rate magnitude
    Cs2PrRatio_3D : ndarray
        Turbulent Cs^2/Pr_t field
    qz_sfc : ndarray
        Surface heat flux

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction
    """

    # Initialize array for vertical flux with correct dimensions
    qz = jnp.zeros_like(S)

    # Interior points for vertical flux (on w-nodes)
    qz = qz.at[:, :, 1:nz - 1].set(
        -2 * (L ** 2) * StagGridAvg(Cs2PrRatio_3D[:, :, :nz - 1]) *
        S[:, :, 1:nz - 1] * dTHdz[:, :, 1:nz - 1]
    )

    # Top boundary condition
    qz = qz.at[:, :, nz - 1].set(0)

    # Bottom boundary condition
    qz = qz.at[:, :, 0].set(qz_sfc)

    return qz
