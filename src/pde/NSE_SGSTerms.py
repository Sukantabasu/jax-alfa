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
File: NSE_SGSTerms.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-6
:Description: computes the SGS stress divergence terms
"""

# ============================================
# Imports
# ============================================

import jax

# Import configuration from namelist
from ..config import Config

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT

# Import derivative functions
from ..operations.Derivatives import Derivxy, Derivz_Generic_uvp, Derivz_Generic_w

# Import SGS models
from ..subgridscale.DynamicSGS_Main import DynamicSGS
from ..subgridscale.StaticSGS_Main import StaticSGS


# =================================================
# Compute Divergence of Stress Tensor
# =================================================

@jax.jit
def DivStress(
        txx, tyy, tzz,
        txy, txz, tyz,
        ZeRo3D,
        kx2, ky2):
    """
    Parameters:
    -----------
    txx, tyy, tzz : ndarray of shape (nx, ny, nz)
        Normal stress components
    txy, txz, tyz : ndarray of shape (nx, ny, nz)
        Shear stress components
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for intermediate calculations

    Returns:
    --------
    tuple containing:
        divtx : ndarray of shape (nx, ny, nz)
            x-component of stress divergence
        divty : ndarray of shape (nx, ny, nz)
            y-component of stress divergence
        divtz : ndarray of shape (nx, ny, nz)
            z-component of stress divergence
    """
    # Compute derivatives in x-direction
    dxtxx = Derivxy(FFT(txx), kx2)
    dxtxy = Derivxy(FFT(txy), kx2)
    dxtxz = Derivxy(FFT(txz), kx2)

    # Compute derivatives in y-direction
    dytyy = Derivxy(FFT(tyy), ky2)
    dytxy = Derivxy(FFT(txy), ky2)
    dytyz = Derivxy(FFT(tyz), ky2)

    # Compute derivatives in z-direction
    # For txz and tyz, use w-grid derivative
    dztxz = Derivz_Generic_w(txz, ZeRo3D)
    dztyz = Derivz_Generic_w(tyz, ZeRo3D)

    # For tzz, use uvp-grid derivative
    dztzz = Derivz_Generic_uvp(tzz, ZeRo3D)

    # Compute divergence components
    divtx = dxtxx + dytxy + dztxz
    divty = dxtxy + dytyy + dztyz
    divtz = dxtxz + dytyz + dztzz

    return divtx, divty, divtz


# =================================================
# Compute stress divergence using dynamic SGS models
# =================================================

@jax.jit
def DivStressDynamicSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        u, v, w, M_sfc_loc, MOSTfunctions,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Compute stress divergence terms using dynamic SGS models.

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
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    divtx, divty, divtz : ndarray of shape (nx, ny, nz)
        Components of stress divergence
    Cs2_1D_avg1, Cs2_1D_avg2 : ndarray of shape (nz)
        Profiles of dynamic Smagorinsky coefficient
    beta1_1D : ndarray of shape (nz)
        Profile of filter width ratio
    dynamicSGSmomentum : tuple
        Complete set of results from dynamic SGS calculation
    """

    # Unpack MOSTfunctions
    (psi2D_m, psi2D_m0, _, _, _, _) = MOSTfunctions

    # Call DynamicSGS to get stresses and other variables
    dynamicSGSmomentum = DynamicSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        u, v, w, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft)

    # Unpack only the variables we need
    txx, tyy, tzz, txy, txz, tyz = dynamicSGSmomentum[0:6]
    Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D = dynamicSGSmomentum[6:9]

    # Compute divergence of stress
    divtx, divty, divtz = DivStress(txx, tyy, tzz,
                                    txy, txz, tyz,
                                    ZeRo3D,
                                    kx2, ky2)

    # Return both the divergence terms and the complete momentum results,
    # so they can be reused for scalar calculations
    return divtx, divty, divtz, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D, dynamicSGSmomentum


# =================================================
# Compute stress divergence using static SGS models
# =================================================

@jax.jit
def DivStressStaticSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        Cs2_3D,
        u, v, M_sfc_loc, MOSTfunctions,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Compute stress divergence terms using static SGS models.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray of shape (nx, ny, nz)
        Derivatives of velocity components in z-direction
    Cs2_3D : ndarray of shape (nx, ny, nz)
        Static Smagorinsky coefficient
    u, v : ndarray of shape (nx, ny, nz)
        Velocity components
    M_sfc_loc : ndarray of shape (nx, ny)
        Near-surface wind speed
    psi2D_m, psi2D_m0 : ndarray of shape (nx, ny)
        Stability correction functions
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated arrays for calculations
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    divtx, divty, divtz : ndarray of shape (nx, ny, nz)
        Components of stress divergence
    staticSGSmomentum : tuple
        Complete set of results from static SGS calculation
    """

    # Unpack MOSTfunctions
    (psi2D_m, psi2D_m0, _, _, _, _) = MOSTfunctions

    # Static SGS model
    staticSGSmomentum = StaticSGS(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        Cs2_3D,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft)

    # Unpack only the stress components we need
    txx, tyy, tzz, txy, txz, tyz = staticSGSmomentum[0:6]

    # Compute divergence of stress
    divtx, divty, divtz = DivStress(txx, tyy, tzz,
                                    txy, txz, tyz,
                                    ZeRo3D,
                                    kx2, ky2)

    # Return both the divergence terms and the complete momentum results,
    # so they can be reused for scalar calculations
    return divtx, divty, divtz, staticSGSmomentum
