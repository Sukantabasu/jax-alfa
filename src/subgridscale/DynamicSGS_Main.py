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
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: dynamic SGS modeling - main code.
              Dispatches between SM (optSgs=1,3) and WL (optSgs=2,4) based on Config.
              optSgs=1: LASDD-SM, optSgs=2: LASDD-WL,
              optSgs=3: LAD-SM, optSgs=4: LAD-WL
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

# Import LASDD models (SM and WL)
from .DynamicSGS_LASDD_SM import LASDD as LASDD_SM
from .DynamicSGS_LASDD_WL import LASDD as LASDD_WL
from .DynamicSGS_ScalarLASDD_SM import ScalarLASDD as ScalarLASDD_SM
from .DynamicSGS_ScalarLASDD_WL import ScalarLASDD as ScalarLASDD_WL

# Import stress functions (SM and WL)
from .SGSStresses_SM import (
    StressesUVPnodes_Dealias   as StressesUVPnodes_Dealias_SM,
    StressesUVPnodes_NoDealias as StressesUVPnodes_NoDealias_SM,
    StressesWnodes_Dealias     as StressesWnodes_Dealias_SM,
    StressesWnodes_NoDealias   as StressesWnodes_NoDealias_SM)
from .SGSStresses_WL import (
    StressesUVPnodes_Dealias   as StressesUVPnodes_Dealias_WL,
    StressesUVPnodes_NoDealias as StressesUVPnodes_NoDealias_WL,
    StressesWnodes_Dealias     as StressesWnodes_Dealias_WL,
    StressesWnodes_NoDealias   as StressesWnodes_NoDealias_WL)

# Import scalar flux functions (SM and WL)
from .ScalarSGSFluxes_SM import (
    ScalarFluxesUVPnodes_Dealias   as ScalarFluxesUVPnodes_Dealias_SM,
    ScalarFluxesUVPnodes_NoDealias as ScalarFluxesUVPnodes_NoDealias_SM,
    ScalarFluxesWnodes_Dealias     as ScalarFluxesWnodes_Dealias_SM,
    ScalarFluxesWnodes_NoDealias   as ScalarFluxesWnodes_NoDealias_SM)
from .ScalarSGSFluxes_WL import (
    ScalarFluxesUVPnodes_Dealias   as ScalarFluxesUVPnodes_Dealias_WL,
    ScalarFluxesUVPnodes_NoDealias as ScalarFluxesUVPnodes_NoDealias_WL,
    ScalarFluxesWnodes_Dealias     as ScalarFluxesWnodes_Dealias_WL,
    ScalarFluxesWnodes_NoDealias   as ScalarFluxesWnodes_NoDealias_WL)


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
    Dispatches to SM variants (optSgs=1,3) or WL variants (optSgs=2,4).

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
        Pre-allocated zero arrays

    Returns:
    --------
    txx, tyy, tzz, txy, txz, tyz : ndarray of shape (nx, ny, nz)
        SGS stress components
    Cs2_1D_avg1, Cs2_1D_avg2 : ndarray of shape (nz)
        1D profiles of SGS model coefficient (two averaging methods)
    beta1_1D : ndarray of shape (nz)
        1D profile of scale-dependence parameter beta1
    u_, v_, w_ : ndarray of shape (nx, ny, nz)
        Interpolated velocity components
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
        # Call LASDD and compute UVP stresses
        # --------------------------------------
        if optSgs in [1, 3]:  # SM variants

            (u_, v_, w_,
             u_hat, v_hat, w_hat,
             u_hatd, v_hatd, w_hatd,
             S_uvp_hat, S_uvp_hatd,
             Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
                LASDD_SM(
                    u, v, w,
                    S11, S22, S33,
                    S12, S13, S23,
                    S_uvp,
                    ZeRo3D))

            Cs2_3D_pad = Dealias1(FFT(Cs2_3D), ZeRo3D_pad_fft)

            (txx, tyy, tzz, txy) = (
                StressesUVPnodes_Dealias_SM(
                    S11_pad, S22_pad, S33_pad, S12_pad,
                    S_uvp_pad,
                    Cs2_3D_pad,
                    ZeRo3D_fft))

        elif optSgs in [2, 4]:  # WL variants

            (u_, v_, w_,
             u_hat, v_hat, w_hat,
             u_hatd, v_hatd, w_hatd,
             S_uvp_hat, S_uvp_hatd,
             Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
                LASDD_WL(
                    u, v, w,
                    S11, S22, S33,
                    S12, S13, S23,
                    ZeRo3D))

            Cs2_3D_pad = Dealias1(FFT(Cs2_3D), ZeRo3D_pad_fft)

            (txx, tyy, tzz, txy) = (
                StressesUVPnodes_Dealias_WL(
                    S11_pad, S22_pad, S33_pad, S12_pad,
                    Cs2_3D_pad,
                    ZeRo3D_fft))

        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS")

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

        S_uvp_pad = S_uvp

        # --------------------------------------
        # Call LASDD and compute UVP stresses
        # --------------------------------------
        if optSgs in [1, 3]:  # SM variants

            (u_, v_, w_,
             u_hat, v_hat, w_hat,
             u_hatd, v_hatd, w_hatd,
             S_uvp_hat, S_uvp_hatd,
             Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
                LASDD_SM(
                    u, v, w,
                    S11, S22, S33,
                    S12, S13, S23,
                    S_uvp,
                    ZeRo3D))

            (txx, tyy, tzz, txy) = (
                StressesUVPnodes_NoDealias_SM(
                    S11, S22, S33, S12,
                    S_uvp,
                    Cs2_3D))

        elif optSgs in [2, 4]:  # WL variants

            (u_, v_, w_,
             u_hat, v_hat, w_hat,
             u_hatd, v_hatd, w_hatd,
             S_uvp_hat, S_uvp_hatd,
             Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D) = (
                LASDD_WL(
                    u, v, w,
                    S11, S22, S33,
                    S12, S13, S23,
                    ZeRo3D))

            (txx, tyy, tzz, txy) = (
                StressesUVPnodes_NoDealias_WL(
                    S11, S22, S33, S12,
                    Cs2_3D))

        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS")

    # ------------------------------------------------------------
    # Compute txz and tyz components
    # ------------------------------------------------------------
    if optDealias == 1:

        (S13_pad, S23_pad,
         S_w, S_w_pad) = (
            StrainsWnodes_Dealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D, ZeRo3D_pad_fft))

        if optSgs in [1, 3]:  # SM variants
            (txz, tyz) = (
                StressesWnodes_Dealias_SM(
                    S13_pad, S23_pad,
                    S_w_pad,
                    Cs2_3D_pad,
                    u, v, M_sfc_loc, psi2D_m, psi2D_m0,
                    ZeRo3D_fft))
        elif optSgs in [2, 4]:  # WL variants
            (txz, tyz) = (
                StressesWnodes_Dealias_WL(
                    S13_pad, S23_pad,
                    Cs2_3D_pad,
                    u, v, M_sfc_loc, psi2D_m, psi2D_m0,
                    ZeRo3D_fft))
        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS")

    else:

        (S13, S23,
         S_w) = (
            StrainsWnodes_NoDealias(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                ZeRo3D))

        S_w_pad = S_w

        if optSgs in [1, 3]:  # SM variants
            (txz, tyz) = (
                StressesWnodes_NoDealias_SM(
                    S13, S23,
                    S_w,
                    Cs2_3D,
                    u, v, M_sfc_loc, psi2D_m, psi2D_m0))
        elif optSgs in [2, 4]:  # WL variants
            (txz, tyz) = (
                StressesWnodes_NoDealias_WL(
                    S13, S23,
                    Cs2_3D,
                    u, v, M_sfc_loc, psi2D_m, psi2D_m0))
        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS")

    return (txx, tyy, tzz, txy, txz, tyz,
            Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D,
            Cs2_3D,
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
        qz_sfc,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    u_, v_, w_ : ndarray of shape (nx, ny, nz)
        Interpolated velocity components
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
    qz_sfc : ndarray of shape (nx, ny)
        Surface sensible heat flux
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated arrays for calculations

    Returns:
    --------
    qx, qy, qz : ndarray of shape (nx, ny, nz)
        SGS scalar flux components
    Cs2PrRatio_1D : ndarray of shape (nz)
        1D profile of SGS coefficient / Pr_t
    beta2_1D : ndarray of shape (nz)
        1D profile of scalar scale-dependence parameter beta2
    """

    # ------------------------------------------------------------
    # Compute scalar SGS model coefficient
    # ------------------------------------------------------------
    if optSgs in [1, 3]:  # SM variants
        (Cs2PrRatio_3D, Cs2PrRatio_1D, beta2_1D) = (
            ScalarLASDD_SM(
                u_, v_, w_,
                u_hat, v_hat, w_hat,
                u_hatd, v_hatd, w_hatd,
                TH,
                dTHdx, dTHdy, dTHdz,
                S_uvp, S_uvp_hat, S_uvp_hatd,
                ZeRo3D))
    elif optSgs in [2, 4]:  # WL variants
        (Cs2PrRatio_3D, Cs2PrRatio_1D, beta2_1D) = (
            ScalarLASDD_WL(
                u_, v_, w_,
                u_hat, v_hat, w_hat,
                u_hatd, v_hatd, w_hatd,
                TH,
                dTHdx, dTHdy, dTHdz,
                ZeRo3D))
    else:
        raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS scalar")

    # ------------------------------------------------------------
    # Compute qx, qy and qz components
    # ------------------------------------------------------------
    if optDealias == 1:

        dTHdx_pad = Dealias1(FFT(dTHdx), ZeRo3D_pad_fft)
        dTHdy_pad = Dealias1(FFT(dTHdy), ZeRo3D_pad_fft)
        dTHdz_pad = Dealias1(FFT(dTHdz), ZeRo3D_pad_fft)

        Cs2PrRatio_3D_pad = Dealias1(FFT(Cs2PrRatio_3D), ZeRo3D_pad_fft)

        if optSgs in [1, 3]:  # SM variants
            (qx, qy) = (
                ScalarFluxesUVPnodes_Dealias_SM(
                    dTHdx_pad, dTHdy_pad,
                    S_uvp_pad,
                    Cs2PrRatio_3D_pad,
                    ZeRo3D_fft))
            qz = (
                ScalarFluxesWnodes_Dealias_SM(
                    dTHdz_pad,
                    S_w_pad,
                    Cs2PrRatio_3D_pad,
                    qz_sfc,
                    ZeRo3D_fft))
        elif optSgs in [2, 4]:  # WL variants
            (qx, qy) = (
                ScalarFluxesUVPnodes_Dealias_WL(
                    dTHdx_pad, dTHdy_pad,
                    Cs2PrRatio_3D_pad,
                    ZeRo3D_fft))
            qz = (
                ScalarFluxesWnodes_Dealias_WL(
                    dTHdz_pad,
                    Cs2PrRatio_3D_pad,
                    qz_sfc,
                    ZeRo3D_fft))
        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS scalar")

    else:

        if optSgs in [1, 3]:  # SM variants
            (qx, qy) = (
                ScalarFluxesUVPnodes_NoDealias_SM(
                    dTHdx, dTHdy,
                    S_uvp,
                    Cs2PrRatio_3D))
            qz = (
                ScalarFluxesWnodes_NoDealias_SM(
                    dTHdz,
                    S_w,
                    Cs2PrRatio_3D,
                    qz_sfc))
        elif optSgs in [2, 4]:  # WL variants
            (qx, qy) = (
                ScalarFluxesUVPnodes_NoDealias_WL(
                    dTHdx, dTHdy,
                    Cs2PrRatio_3D))
            qz = (
                ScalarFluxesWnodes_NoDealias_WL(
                    dTHdz,
                    Cs2PrRatio_3D,
                    qz_sfc))
        else:
            raise ValueError(f"Unsupported optSgs={optSgs} for dynamic SGS scalar")

    return (qx, qy, qz,
            Cs2PrRatio_3D,
            Cs2PrRatio_1D, beta2_1D)
