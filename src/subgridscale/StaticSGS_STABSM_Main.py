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
File: StaticSGS_STABSM_Main.py
================================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-22
:Description: Main dispatcher for the stability-dependent Smagorinsky model
              (STAB-SM, optSgs=5).

              StaticSGS_STABSM  — computes all 6 SGS stress components and
                                   returns pre-computed Lambda/fhS fields for
                                   scalar and moisture reuse.

              StaticSGSscalar_STABSM — computes SGS scalar (or moisture)
                                        fluxes using pre-computed fields from
                                        the momentum step.

              Statistics note: Lambda_uvp2_1D / L^2 is returned as the
              effective-Cs^2 profile so that the statistics output contains
              the actual (z-varying) static mixing-length coefficient rather
              than zeros.
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1

# Import strain rates functions
from .StrainRates import StrainsUVPnodes_Dealias, StrainsWnodes_Dealias
from .StrainRates import StrainsUVPnodes_NoDealias, StrainsWnodes_NoDealias

# Import STAB-SM stress functions
from .SGSStresses_STABSM import (
    StressesUVPnodes_Dealias_STABSM,
    StressesUVPnodes_NoDealias_STABSM,
    StressesWnodes_Dealias_STABSM,
    StressesWnodes_NoDealias_STABSM)

# Import STAB-SM scalar flux functions
from .ScalarSGSFluxes_STABSM import (
    ScalarFluxes_Dealias_STABSM,
    ScalarFluxes_NoDealias_STABSM)


# ============================================================
# STAB-SM: compute all SGS stresses
# ============================================================

@jax.jit
def StaticSGS_STABSM(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        dTHdz,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Computes all 6 SGS stress components using the stability-dependent
    Smagorinsky model and returns the pre-computed Lambda and fhS fields
    needed by StaticSGSscalar_STABSM.

    Parameters:
    -----------
    dudx .. dwdz : ndarray (nx, ny, nz)
        Velocity gradient tensor components
    dTHdz : ndarray (nx, ny, nz)
        Vertical potential temperature gradient (W nodes; for Richardson number)
    u, v : ndarray (nx, ny, nz)
        Velocity components (for wall model)
    M_sfc_loc : ndarray (nx, ny)
        Near-surface wind speed
    psi2D_m, psi2D_m0 : ndarray (nx, ny)
        Stability correction functions for wall model
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays

    Returns:
    --------
    txx, tyy, tzz, txy, txz, tyz : ndarray (nx, ny, nz)
        SGS stress components
    S_uvp, S_uvp_pad : ndarray (nx, ny, nz)
        Strain rate magnitude at UVP nodes
    S_w, S_w_pad : ndarray (nx, ny, nz)
        Strain rate magnitude at W nodes
    Lambda_uvp2_3D : ndarray (nx, ny, nz)
        3-D mixing-length field at UVP nodes (for scalar reuse)
    Lambda_w2_3D : ndarray (nx, ny, nz)
        3-D mixing-length field at W nodes (for scalar reuse)
    fhS_uvp : ndarray (nx, ny, nz)
        fh * |S| at UVP nodes (for scalar reuse)
    fhS_w : ndarray (nx, ny, nz)
        fh * |S| at W nodes (for scalar reuse)
    Lambda_uvp2_1D : ndarray (nz,)
        Horizontal-mean raw Lambda (mixing-length squared) at UVP nodes.
        Main.py divides by L**2 before storing as Cs2_1D_avg* so that
        the statistics output holds effective Cs^2 = Lambda / L^2.
    """

    if optDealias == 1:

        # Strain rates at UVP nodes
        (S11, S22, S33,
         S12, S13, S23,
         S_uvp,
         S11_pad, S22_pad, S33_pad,
         S12_pad, S13_pad, S23_pad,
         S_uvp_pad) = (
            StrainsUVPnodes_Dealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D, ZeRo3D_pad_fft))

        # UVP stresses + Lambda/fhS for scalar
        (txx, tyy, tzz, txy,
         Lambda_uvp2_3D, fhS_uvp,
         Lambda_uvp2_1D) = (
            StressesUVPnodes_Dealias_STABSM(
                S11_pad, S22_pad, S33_pad, S12_pad,
                S_uvp, S_uvp_pad,
                dTHdz,
                ZeRo3D,
                ZeRo3D_fft,
                ZeRo3D_pad_fft))

        # Strain rates at W nodes
        (S13_pad, S23_pad,
         S_w, S_w_pad) = (
            StrainsWnodes_Dealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D, ZeRo3D_pad_fft))

        # W stresses + Lambda/fhS for scalar
        (txz, tyz,
         Lambda_w2_3D, fhS_w) = (
            StressesWnodes_Dealias_STABSM(
                S13_pad, S23_pad,
                S_w, S_w_pad,
                dTHdz,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0,
                ZeRo3D_fft,
                ZeRo3D_pad_fft))

    else:

        # Strain rates at UVP nodes
        (S11, S22, S33,
         S12, S13, S23,
         S_uvp) = (
            StrainsUVPnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        S_uvp_pad = S_uvp  # dummy alias

        # UVP stresses
        (txx, tyy, tzz, txy,
         Lambda_uvp2_3D, fhS_uvp,
         Lambda_uvp2_1D) = (
            StressesUVPnodes_NoDealias_STABSM(
                S11, S22, S33, S12,
                S_uvp,
                dTHdz,
                ZeRo3D))

        # Strain rates at W nodes
        (S13, S23,
         S_w) = (
            StrainsWnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        S_w_pad = S_w  # dummy alias

        # W stresses
        (txz, tyz,
         Lambda_w2_3D, fhS_w) = (
            StressesWnodes_NoDealias_STABSM(
                S13, S23,
                S_w,
                dTHdz,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0))

    return (txx, tyy, tzz, txy, txz, tyz,
            S_uvp, S_uvp_pad,
            S_w, S_w_pad,
            Lambda_uvp2_3D, Lambda_w2_3D,
            fhS_uvp, fhS_w,
            Lambda_uvp2_1D)


# ============================================================
# STAB-SM: compute SGS scalar (or moisture) fluxes
# ============================================================

@jax.jit
def StaticSGSscalar_STABSM(
        Lambda_uvp2_3D,
        Lambda_w2_3D,
        fhS_uvp,
        fhS_w,
        dTHdx, dTHdy, dTHdz,
        qz_sfc,
        ZeRo3D_fft,
        ZeRo3D_pad_fft):
    """
    Computes SGS scalar fluxes using pre-computed Lambda and fhS fields
    from the momentum step.  The same function handles both potential
    temperature and moisture (caller passes the appropriate gradients and
    surface flux).

    Parameters:
    -----------
    Lambda_uvp2_3D : ndarray (nx, ny, nz)
        3-D mixing-length field at UVP nodes
    Lambda_w2_3D : ndarray (nx, ny, nz)
        3-D mixing-length field at W nodes
    fhS_uvp : ndarray (nx, ny, nz)
        fh * |S| at UVP nodes
    fhS_w : ndarray (nx, ny, nz)
        fh * |S| at W nodes
    dTHdx, dTHdy : ndarray (nx, ny, nz)
        Horizontal scalar gradients (UVP nodes)
    dTHdz : ndarray (nx, ny, nz)
        Vertical scalar gradient (W nodes)
    qz_sfc : ndarray (nx, ny)
        Surface scalar flux
    ZeRo3D_fft : ndarray
        Pre-allocated FFT zero array for Dealias2
    ZeRo3D_pad_fft : ndarray
        Pre-allocated padded FFT zero array for Dealias1

    Returns:
    --------
    qx, qy, qz : ndarray (nx, ny, nz)
        SGS scalar flux components
    """

    if optDealias == 1:
        qx, qy, qz = ScalarFluxes_Dealias_STABSM(
            Lambda_uvp2_3D, Lambda_w2_3D,
            fhS_uvp, fhS_w,
            dTHdx, dTHdy, dTHdz,
            qz_sfc,
            ZeRo3D_fft, ZeRo3D_pad_fft)
    else:
        qx, qy, qz = ScalarFluxes_NoDealias_STABSM(
            Lambda_uvp2_3D, Lambda_w2_3D,
            fhS_uvp, fhS_w,
            dTHdx, dTHdy, dTHdz,
            qz_sfc)

    return qx, qy, qz
