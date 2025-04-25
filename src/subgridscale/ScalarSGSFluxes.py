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
==============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-7
:Description: computes SGS scalar fluxes for the eddy-diffusivity model:
              q_i = -2(L^2) * Cs^2/Pr_t * |S| * (∂TH/∂x_i)
              where Cs^2/Pr_t is the model coefficient, |S| is the strain rate
              magnitude, and ∂TH/∂x_i is the temperature gradient
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


# ============================================================
# Compute SGS scalar fluxes on UVP nodes with dealiasing
# ============================================================

@jax.jit
def ScalarFluxesUVPnodes_Dealias(
        dTHdx_pad, dTHdy_pad,
        S_pad, Cs2PrRatio_3D_pad,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdx_pad, dTHdy_pad : ndarray
        Dealiased temperature gradients at UVP nodes (u, v, pressure grid points)
    S_pad : ndarray
        Dealiased strain rate magnitude at UVP nodes
    Cs2PrRatio_3D_pad : ndarray
        Dealiased turbulent Prandtl number field (Cs^2/Pr_t)
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for FFT operations

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components at UVP nodes in x and y directions
        computed using the eddy-diffusivity model: q_i = -2(L^2) * Cs^2/Pr_t * |S| * (∂TH/∂x_i)

    Notes:
    ------
    This function applies dealiasing techniques to avoid aliasing errors in the
    computation of SGS fluxes. The dealiasing is performed using the 3/2 rule
    implemented in the Dealias2 function.
    """

    # Compute SGS scalar fluxes at UVP nodes
    L_term = -2 * (L ** 2)
    qx_pad = L_term * Cs2PrRatio_3D_pad * S_pad * dTHdx_pad
    qy_pad = L_term * Cs2PrRatio_3D_pad * S_pad * dTHdy_pad

    # Set top boundary conditions
    qx_pad = qx_pad.at[:, :, nz - 1].set(0)
    qy_pad = qy_pad.at[:, :, nz - 1].set(0)

    # Apply dealiasing to horizontal fluxes
    qx = Dealias2(FFT_pad(qx_pad), ZeRo3D_fft)
    qy = Dealias2(FFT_pad(qy_pad), ZeRo3D_fft)

    return qx, qy


# ============================================================
# Compute SGS scalar fluxes on UVP nodes without dealiasing
# ============================================================

@jax.jit
def ScalarFluxesUVPnodes_NoDealias(
        dTHdx, dTHdy,
        S, Cs2PrRatio_3D):
    """
    Parameters:
    -----------
    dTHdx, dTHdy : ndarray
        Temperature gradients at UVP nodes (u, v, pressure grid points)
    S : ndarray
        Strain rate magnitude at UVP nodes
    Cs2PrRatio_3D : ndarray
        Turbulent Prandtl number field (Cs^2/Pr_t)

    Returns:
    --------
    qx, qy : ndarray
        SGS scalar flux components at UVP nodes in x and y directions
        computed using the eddy-diffusivity model: q_i = -2(L^2) * Cs^2/Pr_t * |S| * (∂TH/∂x_i)

    Notes:
    ------
    This is the direct computation version without dealiasing. It's more
    computationally efficient but may introduce aliasing errors in non-linear
    operations. Used when optDealias = 0 in the configuration.
    """

    # Compute SGS scalar fluxes at UVP nodes
    L_term = -2 * (L ** 2)
    qx = L_term * Cs2PrRatio_3D * S * dTHdx
    qy = L_term * Cs2PrRatio_3D * S * dTHdy

    # Set top boundary conditions
    qx = qx.at[:, :, nz - 1].set(0)
    qy = qy.at[:, :, nz - 1].set(0)

    return qx, qy


# ============================================================
# Compute SGS scalar fluxes on W nodes with dealiasing
# ============================================================

@jax.jit
def ScalarFluxesWnodes_Dealias(
        dTHdz_pad,
        S_pad, Cs2PrRatio_3D_pad,
        SHFX,
        ZeRo3D_fft):
    """
    Parameters:
    -----------
    dTHdz_pad : ndarray
        Dealiased temperature gradient in z-direction at W nodes (vertical velocity grid points)
    S_pad : ndarray
        Dealiased strain rate magnitude at W nodes
    Cs2PrRatio_3D_pad : ndarray
        Dealiased turbulent Prandtl number field (Cs^2/Pr_t)
    SHFX : ndarray
        Surface heat flux, shape (nx, ny), represents the prescribed boundary condition
    ZeRo3D_fft : ndarray
        Pre-allocated zero array for FFT operations

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction at W nodes computed using
        the eddy-diffusivity model, with special handling for staggered grid.

    Notes:
    ------
    This function computes the vertical flux component on the W-grid (staggered grid
    for vertical velocity). It applies the StagGridAvg function to properly average
    the Cs2PrRatio_3D_pad values to the W-grid points. The bottom boundary condition
    is set using the prescribed surface heat flux (SHFX).
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

    # Bottom boundary condition - prescribed surface flux
    qz = qz.at[:, :, 0].set(SHFX)

    return qz


# ============================================================
# Compute SGS scalar fluxes on W nodes without dealiasing
# ============================================================

@jax.jit
def ScalarFluxesWnodes_NoDealias(
        dTHdz,
        S, Cs2PrRatio_3D,
        SHFX):
    """
    Parameters:
    -----------
    dTHdz : ndarray
        Temperature gradient in z-direction at W nodes (vertical velocity grid points)
    S : ndarray
        Strain rate magnitude at W nodes
    Cs2PrRatio_3D : ndarray
        Turbulent Prandtl number field (Cs^2/Pr_t)
    SHFX : ndarray
        Surface heat flux, shape (nx, ny), represents the prescribed boundary condition

    Returns:
    --------
    qz : ndarray
        SGS scalar flux component in z-direction at W nodes computed using
        the eddy-diffusivity model, with special handling for staggered grid.

    Notes:
    ------
    This is the direct computation version without dealiasing. It performs the same
    calculation as ScalarFluxesWnodes_Dealias but without the dealiasing steps.
    The bottom boundary condition is prescribed by SHFX, and the top boundary
    is set to zero to enforce the free-slip condition at the domain top.
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

    # Bottom boundary condition - prescribed surface flux
    qz = qz.at[:, :, 0].set(SHFX)

    return qz
