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
File: SCL_SGSTerms.py
=====================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: computes the SGS scalar flux divergence terms
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
from ..operations.Derivatives import Derivxy, Derivz_Generic_w

# Import SGS models
from ..subgridscale.DynamicSGS_Main import DynamicSGSscalar
from ..subgridscale.StaticSGS_Main import StaticSGSscalar


# =================================================
# Compute Divergence of Scalar Flux
# =================================================

@jax.jit
def DivFlux(
        qx, qy, qz,
        ZeRo3D,
        kx2, ky2):
    """
    Parameters:
    -----------
    qx, qy : ndarray of shape (nx, ny, nz)
        Horizontal scalar flux components
    qz : ndarray of shape (nx, ny, nz)
        Vertical scalar flux component
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for intermediate calculations

    Returns:
    --------
    divq : ndarray of shape (nx, ny, nz)
        Divergence of scalar flux
    """
    # Compute derivatives in x-direction
    dxqx = Derivxy(FFT(qx), kx2)

    # Compute derivatives in y-direction
    dyqy = Derivxy(FFT(qy), ky2)

    # Compute derivatives in z-direction
    # For qz, use w-grid derivative
    dzqz = Derivz_Generic_w(qz, ZeRo3D)

    # Compute divergence
    divq = dxqx + dyqy + dzqz

    return divq


# ================================================
# Compute flux divergence using dynamic SGS models
# ================================================

@jax.jit
def DivFluxDynamicSGS(
        dynamicSGSmomentum,
        TH,
        dTHdx, dTHdy, dTHdz,
        SHFX,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Compute flux divergence using dynamic SGS models.

    Parameters:
    -----------
    dynamicSGSmomentum : tuple
        Results from dynamic SGS momentum calculations,
        containing filtered velocity components and strain rates
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature
    dTHdx, dTHdy, dTHdz : ndarray of shape (nx, ny, nz)
        Gradients of potential temperature
    SHFX : ndarray of shape (nx, ny)
        Surface sensible heat flux
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays for calculations
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    divq : ndarray of shape (nx, ny, nz)
        Divergence of scalar flux
    Cs2PrRatio_1D : ndarray of shape (nz)
        1D profile of dynamic SGS coefficient (Cs2/Pr)
    beta2_1D : ndarray of shape (nz)
        1D profile of beta coefficient for scalar SGS model
    """

    # Unpack the dynamicSGSmomentum tuple
    (u_, v_, w_,
     u_hat, v_hat, w_hat,
     u_hatd, v_hatd, w_hatd,
     S_uvp, S_uvp_pad,
     S_w, S_w_pad,
     S_uvp_hat, S_uvp_hatd) = dynamicSGSmomentum

    (qx, qy, qz,
     Cs2PrRatio_1D, beta2_1D) = (
        DynamicSGSscalar(
            u_, v_, w_,
            u_hat, v_hat, w_hat,
            u_hatd, v_hatd, w_hatd,
            S_uvp, S_uvp_pad,
            S_w, S_w_pad,
            S_uvp_hat, S_uvp_hatd,
            TH,
            dTHdx, dTHdy, dTHdz,
            SHFX,
            ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft))

    # Compute divergence of flux
    divq = DivFlux(qx, qy, qz,
                   ZeRo3D,
                   kx2, ky2)

    return qz, divq, Cs2PrRatio_1D, beta2_1D


# ======================================================
# Compute scalar flux divergence using static SGS models
# ======================================================

@jax.jit
def DivFluxStaticSGS(
        staticSGSmomentum,
        Cs2PrRatio_3D,
        dTHdx, dTHdy, dTHdz,
        SHFX,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
        kx2, ky2):
    """
    Compute scalar flux divergence using static SGS models.

    Parameters:
    -----------
    staticSGSmomentum : tuple
        Results from static SGS momentum calculations, containing strain rates
    Cs2PrRatio_3D : ndarray of shape (nx, ny, nz)
        Static coefficient for scalar SGS model
    dTHdx, dTHdy, dTHdz : ndarray of shape (nx, ny, nz)
        Gradients of potential temperature
    SHFX : ndarray of shape (nx, ny)
        Surface heat flux
    ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft : ndarray
        Pre-allocated zero arrays for calculations
    kx2, ky2 : ndarray
        Wavenumber arrays for spectral derivatives

    Returns:
    --------
    divq : ndarray of shape (nx, ny, nz)
        Divergence of scalar flux
    """

    # Unpack the staticSGSmomentum tuple
    (S_uvp, S_uvp_pad,
     S_w, S_w_pad) = staticSGSmomentum

    # Static SGS model for scalar
    qx, qy, qz = (
        StaticSGSscalar(
            S_uvp, S_uvp_pad,
            S_w, S_w_pad,
            Cs2PrRatio_3D,
            dTHdx, dTHdy, dTHdz,
            SHFX,
            ZeRo3D_fft, ZeRo3D_pad_fft))

    # Compute divergence of flux
    divq = DivFlux(qx, qy, qz,
                   ZeRo3D,
                   kx2, ky2)

    return qz, divq
