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
File: StaticSGS_Main.py
========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: static SGS modeling - main code
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

# Import stress functions
from .SGSStresses import StressesUVPnodes_Dealias, StressesWnodes_Dealias
from .SGSStresses import StressesUVPnodes_NoDealias, StressesWnodes_NoDealias

# Import scalar flux functions
from .ScalarSGSFluxes import (ScalarFluxesUVPnodes_Dealias,
                              ScalarFluxesWnodes_Dealias)
from .ScalarSGSFluxes import (ScalarFluxesUVPnodes_NoDealias,
                              ScalarFluxesWnodes_NoDealias)


# ============================================================
# Static SGS: compute all the SGS stresses on proper nodes
# ============================================================

@jax.jit
def StaticSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        Cs2_3D,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz : ndarray
        Velocity gradients
    Cs2_3D : ndarray
        Smagorinsky coefficient field
    u, v : ndarray
        Velocity components for wall model
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays

    Returns:
    --------
    txx, tyy, tzz, txy, txz, tyz : ndarray
        SGS stress components
    S_uvp, S_uvp_pad, S_w, S_w_pad : ndarray
        Strain rate fields
    """

    # ----------------------------------------
    # Compute txx, tyy, tzz and txy components
    # ----------------------------------------
    if optDealias == 1:

        # --------------------------------------
        # Compute strain rates
        # --------------------------------------
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

        # --------------------------------------
        # Dealias Cs2 field
        # --------------------------------------
        Cs2_3D_pad = Dealias1(FFT(Cs2_3D), ZeRo3D_pad_fft)

        # --------------------------------------
        # Compute SGS stresses
        # --------------------------------------
        (txx, tyy, tzz, txy) = (
            StressesUVPnodes_Dealias(
                S11_pad, S22_pad, S33_pad, S12_pad,
                S_uvp_pad,
                Cs2_3D_pad,
                ZeRo3D_fft))

    else:

        # --------------------------------------
        # Compute strain rates
        # --------------------------------------
        (S11, S22, S33,
         S12, S13, S23,
         S_uvp) = (
            StrainsUVPnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        # create a dummy variable for passing
        S_uvp_pad = S_uvp

        # --------------------------------------
        # Compute SGS stresses
        # --------------------------------------
        (txx, tyy, tzz, txy) = (
            StressesUVPnodes_NoDealias(
                S11, S22, S33, S12,
                S_uvp,
                Cs2_3D))

    # ------------------------------------------------------------
    # Compute txz, tyz components
    # ------------------------------------------------------------
    if optDealias == 1:

        # --------------------------------------
        # Compute strain rates
        # --------------------------------------
        (S13_pad, S23_pad,
         S_w_pad) = (
            StrainsWnodes_Dealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D, ZeRo3D_pad_fft))

        # create a dummy variable for passing
        S_w = S_w_pad

        # --------------------------------------
        # Compute SGS stresses
        # --------------------------------------
        (txz, tyz) = (
            StressesWnodes_Dealias(
                S13_pad, S23_pad,
                S_w_pad,
                Cs2_3D_pad,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0,
                ZeRo3D_fft))

    else:

        # --------------------------------------
        # Compute strain rates
        # --------------------------------------
        (S13, S23,
         S_w) = (
            StrainsWnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        # create a dummy variable for passing
        S_w_pad = S_w

        # --------------------------------------
        # Compute SGS stresses
        # --------------------------------------
        (txz, tyz) = (
            StressesWnodes_NoDealias(
                S13, S23,
                S_w,
                Cs2_3D,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0))

    # Return stresses along with strain rates for scalar calculations
    return (txx, tyy, tzz, txy, txz, tyz,
            S_uvp, S_uvp_pad,
            S_w, S_w_pad)


# =====================================================
# Static SGS: compute scalar SGS fluxes on proper nodes
# =====================================================

@jax.jit
def StaticSGSscalar(
        S_uvp, S_uvp_pad,
        S_w, S_w_pad,
        Cs2PrRatio_3D,
        dTHdx, dTHdy, dTHdz,
        qz_sfc,
        ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    S_uvp, S_uvp_pad : ndarray
        Strain rate magnitudes at UVP nodes
    S_w, S_w_pad : ndarray
        Strain rate magnitudes at W nodes
    Cs2PrRatio_3D : ndarray
        Cs^2/Pr_t field
    dTHdx, dTHdy, dTHdz : ndarray
        Potential temperature gradients
    qz_sfc : ndarray
        Surface heat flux
    ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays

    Returns:
    --------
    qx, qy, qz : ndarray
        SGS scalar flux components
    """

    # ------------------------------------------------------------
    # Compute qx, qy and qz components
    # ------------------------------------------------------------
    if optDealias == 1:

        dTHdx_pad = Dealias1(FFT(dTHdx), ZeRo3D_pad_fft)
        dTHdy_pad = Dealias1(FFT(dTHdy), ZeRo3D_pad_fft)
        dTHdz_pad = Dealias1(FFT(dTHdz), ZeRo3D_pad_fft)

        Cs2PrRatio_3D_pad = Dealias1(FFT(Cs2PrRatio_3D), ZeRo3D_pad_fft)

        # Compute fluxes at UVP nodes
        (qx, qy) = (
            ScalarFluxesUVPnodes_Dealias(
                dTHdx_pad, dTHdy_pad,
                S_uvp_pad,
                Cs2PrRatio_3D_pad,
                ZeRo3D_fft))

        # Compute flux on W nodes
        qz = (
            ScalarFluxesWnodes_Dealias(
                dTHdz_pad,
                S_w_pad,
                Cs2PrRatio_3D_pad,
                qz_sfc,
                ZeRo3D_fft))

    else:

        # Compute fluxes at UVP nodes
        (qx, qy) = (
            ScalarFluxesUVPnodes_NoDealias(
                dTHdx, dTHdy,
                S_uvp,
                Cs2PrRatio_3D))

        # Compute flux on W nodes
        qz = (
            ScalarFluxesWnodes_NoDealias(
                dTHdz,
                S_w,
                Cs2PrRatio_3D,
                qz_sfc))

    return qx, qy, qz
