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
File: SCL_SGSTerms_STABSM.py
==============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-22
:Description: PDE wrapper — scalar flux divergence for STAB-SM (optSgs=5).
              Reuses Lambda and fhS fields computed during the momentum step.
              Called for both potential temperature and moisture.
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import derived variables
from ..config.DerivedVars import *

# Import flux divergence helper (shared with other SGS modules)
from .SCL_SGSTerms import DivFlux

# Import STAB-SM scalar function
from ..subgridscale.StaticSGS_STABSM_Main import StaticSGSscalar_STABSM


# ============================================================
# Scalar flux divergence — STAB-SM
# ============================================================

@jax.jit
def DivFluxStaticSGS_STABSM(
        stabsmScalarFields,
        dTHdx, dTHdy, dTHdz,
        SHFX,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Parameters:
    -----------
    stabsmScalarFields : tuple
        (Lambda_uvp2_3D, Lambda_w2_3D, fhS_uvp, fhS_w) from the momentum step
        — indices [10:14] of the stabsmSGSmomentum tuple.
    dTHdx, dTHdy, dTHdz : ndarray (nx, ny, nz)
        Scalar gradients (potential temperature or moisture)
    SHFX : ndarray (nx, ny)
        Surface scalar flux (heat or moisture)
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    qz : ndarray (nx, ny, nz)
        Vertical SGS scalar flux (for statistics)
    divq : ndarray (nx, ny, nz)
        Divergence of SGS scalar flux
    """

    Lambda_uvp2_3D, Lambda_w2_3D, fhS_uvp, fhS_w = stabsmScalarFields

    qx, qy, qz = StaticSGSscalar_STABSM(
        Lambda_uvp2_3D, Lambda_w2_3D,
        fhS_uvp, fhS_w,
        dTHdx, dTHdy, dTHdz,
        SHFX,
        ZeRo3D_fft, ZeRo3D_pad_fft)

    divq = DivFlux(qx, qy, qz, ZeRo3D, kx2, ky2)

    return qz, divq
