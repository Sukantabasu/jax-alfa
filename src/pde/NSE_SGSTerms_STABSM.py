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
File: NSE_SGSTerms_STABSM.py
==============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-22
:Description: PDE wrapper — stress divergence for STAB-SM (optSgs=5).
              Calls StaticSGS_STABSM and then DivStress.
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import derived variables
from ..config.DerivedVars import *

# Import stress divergence helper (shared with other SGS modules)
from .NSE_SGSTerms import DivStress

# Import STAB-SM momentum function
from ..subgridscale.StaticSGS_STABSM_Main import StaticSGS_STABSM


# ============================================================
# Stress divergence — STAB-SM
# ============================================================

@jax.jit
def DivStressStaticSGS_STABSM(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        dTHdz,
        u, v, M_sfc_loc, MOSTfunctions,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Parameters:
    -----------
    dudx .. dwdz : ndarray (nx, ny, nz)
        Velocity gradient tensor components
    dTHdz : ndarray (nx, ny, nz)
        Vertical potential temperature gradient (for Richardson number)
    u, v : ndarray (nx, ny, nz)
        Velocity components (for wall model)
    M_sfc_loc : ndarray (nx, ny)
        Near-surface wind speed
    MOSTfunctions : tuple
        Stability functions from surface flux computation
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    divtx, divty, divtz : ndarray (nx, ny, nz)
        Components of stress divergence
    stabsmSGSmomentum : tuple
        Full results from StaticSGS_STABSM for reuse in scalar step
    """

    (psi2D_m, psi2D_m0, _, _, _, _) = MOSTfunctions

    stabsmSGSmomentum = StaticSGS_STABSM(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        dTHdz,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft)

    txx, tyy, tzz, txy, txz, tyz = stabsmSGSmomentum[0:6]

    divtx, divty, divtz = DivStress(
        txx, tyy, tzz, txy, txz, tyz,
        ZeRo3D, kx2, ky2)

    return divtx, divty, divtz, stabsmSGSmomentum
