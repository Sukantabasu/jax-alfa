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
File: DynamicSGS_Main.py
========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-5
:Description: dynamic SGS modeling
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
from .ScalarSGSFluxes import ScalarFluxesUVPnodes_Dealias, ScalarFluxesWnodes_Dealias
from .ScalarSGSFluxes import ScalarFluxesUVPnodes_NoDealias, ScalarFluxesWnodes_NoDealias

# Import LASDD
from .DynamicSGS_LASDD import LASDD
from .DynamicSGS_ScalarLASDD import ScalarLASDD


# ============================================================
# Dynamic SGS: compute all the SGS stresses on proper nodes
# ============================================================

@jax.jit
def DynamicSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        u, v, w, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Computes all SGS stresses on proper grid nodes using the dynamic model.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in z-direction
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components
    M_sfc_loc : ndarray of shape (nx, ny)
        Near-surface wind speed
    psi2D_m, psi2D_m0 : ndarray of shape (nx, ny)
        Stability correction functions
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated arrays for calculations

    Returns:
    --------
    txx, tyy, tzz, txy, txz, tyz : ndarray of shape (nx, ny, nz)
        SGS stress components
    Cs2_1D_avg1, Cs2_1D_avg2 : ndarray of shape (nz)
        1D profiles of Smagorinsky coefficient (two averaging methods)
    beta1_1D : ndarray of shape (nz)
        1D profile of filter width ratio
    u_, v_, w_ : ndarray of shape (nx, ny, nz)
        Adjusted velocity components
    u_hat, v_hat, w_hat : ndarray of shape (nx, ny, nz)
        Level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : ndarray of shape (nx, ny, nz)
        Level-2 filtered velocity components
    S_uvp, S_uvp_pad : ndarray of shape (nx, ny, nz)
        Strain rate magnitude at UVP nodes and its padded version
    S_w, S_w_pad : ndarray of shape (nx, ny, nz)
        Strain rate magnitude at W nodes and its padded version
    S_uvp_hat, S_uvp_hatd : ndarray of shape (nx, ny, nz)
        Filtered strain rate magnitudes
    """

    # ------------------------------------------------------------
    # Compute txx, tyy, tzz and txy components
    # ------------------------------------------------------------
    if optDealias == 1:

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
        # Call LASDD model
        # --------------------------------------
        (u_, v_, w_,
         u_hat, v_hat, w_hat,
         u_hatd, v_hatd, w_hatd,
         S_uvp_hat, S_uvp_hatd,
         Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
            LASDD(
                u, v, w,
                S11, S22, S33,
                S12, S13, S23,
                S_uvp,
                ZeRo3D))
        # --------------------------------------

        Cs2_3D_pad = Dealias1(FFT(Cs2_3D), ZeRo3D_pad_fft)

        (txx, tyy, tzz, txy) = (
            StressesUVPnodes_Dealias(
                S11_pad, S22_pad, S33_pad, S12_pad,
                S_uvp_pad,
                Cs2_3D_pad,
                ZeRo3D_fft))

    else:

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
        # Call LASDD model
        # --------------------------------------
        (u_, v_, w_,
         u_hat, v_hat, w_hat,
         u_hatd, v_hatd, w_hatd,
         S_uvp_hat, S_uvp_hatd,
         Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
            LASDD(
                u, v, w,
                S11, S22, S33,
                S12, S13, S23,
                S_uvp,
                ZeRo3D))
        # --------------------------------------

        (txx, tyy, tzz, txy) = (
            StressesUVPnodes_NoDealias(
                S11, S22, S33, S12,
                S_uvp,
                Cs2_3D))

    # ------------------------------------------------------------
    # Compute txz and tyz components
    # ------------------------------------------------------------
    if optDealias == 1:

        (S13_pad, S23_pad,
         S_w_pad) = (
            StrainsWnodes_Dealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D, ZeRo3D_pad_fft))

        # create a dummy variable for passing
        S_w = S_w_pad

        (txz, tyz) = (
            StressesWnodes_Dealias(
                S13_pad, S23_pad,
                S_w_pad,
                Cs2_3D_pad,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0,
                ZeRo3D_fft))

    else:

        (S13, S23,
         S_w) = (
            StrainsWnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        # create a dummy variable for passing
        S_w_pad = S_w

        (txz, tyz) = (
            StressesWnodes_NoDealias(
                S13, S23,
                S_w,
                Cs2_3D,
                u, v, M_sfc_loc, psi2D_m, psi2D_m0))

    # Return stresses along with strain rates for scalar calculations
    return (txx, tyy, tzz, txy, txz, tyz,
            Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D,
            u_, v_, w_,
            u_hat, v_hat, w_hat,
            u_hatd, v_hatd, w_hatd,
            S_uvp, S_uvp_pad,
            S_w, S_w_pad,
            S_uvp_hat, S_uvp_hatd)


# ============================================================
# Dynamic SGS: compute scalar SGS fluxes on proper nodes
# ============================================================

@jax.jit
def DynamicSGSscalar(
        u_, v_, w_,
        u_hat, v_hat, w_hat,
        u_hatd, v_hatd, w_hatd,
        S_uvp, S_uvp_pad,
        S_w, S_w_pad,
        S_uvp_hat, S_uvp_hatd,
        TH,
        dTHdx, dTHdy, dTHdz,
        SHFX,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Computes scalar SGS fluxes on proper nodes using the dynamic model.

    Parameters:
    -----------
    u_, v_, w_ : ndarray of shape (nx, ny, nz)
        Adjusted velocity components
    u_hat, v_hat, w_hat : ndarray of shape (nx, ny, nz)
        Level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : ndarray of shape (nx, ny, nz)
        Level-2 filtered velocity components
    S_uvp, S_uvp_pad : ndarray of shape (nx, ny, nz)
        Strain rate magnitude at UVP nodes and its padded version
    S_w, S_w_pad : ndarray of shape (nx, ny, nz)
        Strain rate magnitude at W nodes and its padded version
    S_uvp_hat, S_uvp_hatd : ndarray of shape (nx, ny, nz)
        Filtered strain rate magnitudes
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature
    dTHdx, dTHdy, dTHdz : ndarray of shape (nx, ny, nz)
        Derivatives of potential temperature
    SHFX : ndarray of shape (nx, ny)
        Surface heat flux
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated arrays for calculations

    Returns:
    --------
    qx, qy, qz : ndarray of shape (nx, ny, nz)
        SGS scalar flux components
    Cs2PrRatio_1D : ndarray of shape (nz)
        1D profile of Cs²/Pr ratio
    beta2_1D : ndarray of shape (nz)
        1D profile of filter width ratio for scalar model
    """

    # ------------------------------------------------------------
    # Compute scalar SGS model coefficients
    # ------------------------------------------------------------
    (Cs2PrRatio_3D, Cs2PrRatio_1D, beta2_1D) = (
        ScalarLASDD(
            u_, v_, w_,
            u_hat, v_hat, w_hat,
            u_hatd, v_hatd, w_hatd,
            TH,
            dTHdx, dTHdy, dTHdz,
            S_uvp, S_uvp_hat, S_uvp_hatd,
            ZeRo3D))

    # ------------------------------------------------------------
    # Compute qx, qy and qz components
    # ------------------------------------------------------------
    if optDealias == 1:

        dTHdx_pad = Dealias1(FFT(dTHdx), ZeRo3D_pad_fft)
        dTHdy_pad = Dealias1(FFT(dTHdy), ZeRo3D_pad_fft)
        dTHdz_pad = Dealias1(FFT(dTHdz), ZeRo3D_pad_fft)

        Cs2PrRatio_3D_pad = Dealias1(FFT(Cs2PrRatio_3D), ZeRo3D_pad_fft)

        # Compute fluxes on UVP nodes
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
                SHFX,
                ZeRo3D_fft))

    else:

        # Compute fluxes at UVP nodes
        (qx, qy) = (
            ScalarFluxesUVPnodes_NoDealias(
                dTHdx, dTHdy,
                S_uvp,
                Cs2PrRatio_3D))

        # Compute flux at W nodes
        qz = (
            ScalarFluxesWnodes_NoDealias(
                dTHdz,
                S_w,
                Cs2PrRatio_3D,
                SHFX))

    return (qx, qy, qz,
            Cs2PrRatio_1D, beta2_1D)
