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
File: Derivatives.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: computes derivatives in x, y and z directions
"""


# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *


# ============================================================
#  Compute spatial derivatives
# ============================================================

@jax.jit
def velocityGradients(
        u, v, w,
        u_fft, v_fft, w_fft,
        kx2, ky2,
        ustar, M_sfc_loc, MOSTfunctions, ZeRo3D):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components in x, y, and z directions in physical space
    u_fft, v_fft, w_fft : ndarray of shape (nx, ny//2 + 1, nz)
        Pre-computed Fourier transforms of the velocity components
    kx2, ky2 : ndarray of shape (nx, ny//2 + 1, nz)
        Pre-computed wavenumber arrays for spectral derivatives
    M_sfc_loc : ndarray of shape (nx, ny)
        Near-surface wind speed used for boundary conditions
    ustar : ndarray of shape (nx, ny)
        Friction velocity for boundary condition calculations
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for storing derivative results

    Returns:
    --------
    dudx, dvdx, dwdx :
        x-derivatives of the velocity components
    dudy, dvdy, dwdy :
        y-derivatives of the velocity components
    dudz, dvdz, dwdz :
        z-derivatives of the velocity components

    Notes:
    ------
    - Horizontal derivatives (x, y) are computed using spectral methods via `Derivxy`
    - Vertical derivatives (z) are computed using finite differences via `Derivz_M`
    - Boundary conditions for vertical derivatives are handled in `Derivz_M`
    """

    # X derivatives
    dudx, dvdx, dwdx = (Derivxy(u_fft, kx2),
                        Derivxy(v_fft, kx2),
                        Derivxy(w_fft, kx2))

    # Y derivatives
    dudy, dvdy, dwdy = (Derivxy(u_fft, ky2),
                        Derivxy(v_fft, ky2),
                        Derivxy(w_fft, ky2))

    # unpack MOST functions
    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    # Z derivatives
    dudz, dvdz, dwdz = Derivz_M(u, v, w, ustar, M_sfc_loc, fi2D_m, ZeRo3D)

    # Return all derivatives
    return dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz


@jax.jit
def potentialTemperatureGradients(
        TH, TH_fft,
        kx2, ky2,
        ustar, qz_sfc, MOSTfunctions, ZeRo3D):
    """
    Parameters:
    -----------
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature in physical space
    TH_fft : ndarray of shape (nx, ny//2 + 1, nz)
        Pre-computed Fourier transforms of potential temperature
    kx2, ky2 : ndarray of shape (nx, ny//2 + 1, nz)
        Pre-computed wavenumber arrays for spectral derivatives
    fi2D_h : ndarray of shape (nx, ny)
        Normalized gradient function for Monin-Obukhov similarity
    qz_sfc : ndarray of shape (nx, ny)
        Surface sensible heat flux, unit: K m/s
    ustar : ndarray of shape (nx, ny)
        Friction velocity for boundary condition calculations
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array for storing derivative results

    Returns:
    --------
    dTHdx, dTHdy, dTHdz :
        x,y,z-derivatives of potential temperature

    Notes:
    ------
    - Horizontal derivatives (x, y) are computed using spectral methods via `Derivxy`
    - Vertical derivatives (z) are computed using finite differences via `Derivz_TH`
    - Boundary conditions for vertical derivatives are handled in `Derivz_TH`
    """

    # X derivatives
    dTHdx = Derivxy(TH_fft, kx2)

    # Y derivatives
    dTHdy = Derivxy(TH_fft, ky2)

    # unpack MOST functions
    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    # Z derivatives
    dTHdz = Derivz_TH(TH, ustar, qz_sfc, fi2D_h, ZeRo3D)

    # Return all derivatives
    return dTHdx, dTHdy, dTHdz


# ============================================================
#  Compute spectral derivatives in x or y direction
# ============================================================

@jax.jit
def Derivxy(F_fft, kxy2):
    """
    Parameters:
    -----------
    F_fft : ndarray with shape (nx, ny//2 + 1, nz)
        Fourier-transformed 3D field.
    kxy2 : ndarray shape (nx, ny//2 + 1, nz)
        Pre-computed wavenumbers (use kx2 for dudx; ky2 for dudy)

    Returns:
    --------
    dFdxy : ndarray
        x- or y-derivative field

    Notes:
    ------
    - Nyquist frequencies are explicitly set to zero
    """
    # Compute derivative in Fourier space and transform back
    dFdxy = jnp.fft.irfft2(1j * kxy2 * F_fft, axes=(0, 1), s=(nx, ny))

    return dFdxy


# ============================================================
#  Finite difference-based vertical derivatives for velocity
# ============================================================

@jax.jit
def Derivz_M(u, v, w, ustar, M_sfc_loc, fi2D_m, ZeRo3D):
    """
    Parameters:
    -----------
    u : ndarray of shape (nx, ny, nz)
        Longitudinal velocity component
    v : ndarray of shape (nx, ny, nz)
        Lateral velocity component
    w : ndarray of shape (nx, ny, nz)
        Vertical velocity component
    M : ndarray of shape (nx, ny)
        Near-surface wind speed
    fi2D : ndarray  of shape (nx, ny)
        Normalized gradient function
    ustar : ndarray of shape (nx, ny)
        Friction velocity

    Returns:
    --------
    dudz : Vertical derivative of u
    dvdz : Vertical derivative of v
    dwdz : Vertical derivative of w
    """

    # Initialize arrays with zeros
    dudz = ZeRo3D.copy()
    dvdz = ZeRo3D.copy()
    dwdz = ZeRo3D.copy()

    # Compute interior derivatives using central differences
    dudz = dudz.at[:, :, 1:nz - 1].set(jnp.diff(u[:, :, 0:nz - 1], axis=2) * idz)
    dvdz = dvdz.at[:, :, 1:nz - 1].set(jnp.diff(v[:, :, 0:nz - 1], axis=2) * idz)

    # Bottom boundary conditions using Monin-Obukhov similarity
    dudz = dudz.at[:, :, 0].set(
        fi2D_m * ustar * (u[:, :, 0] + Ugal) / (M_sfc_loc * vonk * 0.5 * dz)
    )
    dvdz = dvdz.at[:, :, 0].set(
        fi2D_m * ustar * v[:, :, 0] / (M_sfc_loc * vonk * 0.5 * dz)
    )

    # Vertical velocity derivatives
    dwdz = dwdz.at[:, :, 0:nz - 1].set(jnp.diff(w, axis=2) * idz)
    dwdz = dwdz.at[:, :, nz - 1].set(0.0)  # Top boundary condition

    return dudz, dvdz, dwdz


# ============================================================
#  Vertical derivatives for temperature
# ============================================================

@jax.jit
def Derivz_TH(TH, ustar, qz_sfc, fi2D_h, ZeRo3D):
    """
    Parameters:
    -----------
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature
    fi2D_h : ndarray of shape (nx, ny)
        Normalized gradient function for heat
    qz_sfc : ndarray of shape (nx, ny)
        Surface sensible heat flux, unit: K m/s
    ustar : ndarray of shape (nx, ny)
        Friction velocity

    Returns:
    --------
    ndarray
        dTHdz : Vertical derivative of potential temperature
    """

    # Initialize array with zeros
    dTHdz = ZeRo3D.copy()

    # Compute interior derivatives
    dTHdz = dTHdz.at[:, :, 1:nz].set(jnp.diff(TH, axis=2) * idz)

    # Bottom boundary condition using Monin-Obukhov similarity
    dTHdz = dTHdz.at[:, :, 0].set(
        fi2D_h * (-qz_sfc / ustar) / (vonk * 0.5 * dz)
    )

    return dTHdz


# ============================================================
#  Vertical derivatives for a generic variable on uvp nodes
# ============================================================

@jax.jit
def Derivz_Generic_uvp(F, ZeRo3D):
    """
    Parameters:
    -----------
    F : ndarray of shape (nx, ny, nz)
        Generic variable defined on uvp nodes

    Returns:
    --------
    ndarray
        dFdz : Vertical derivative of F
    """

    # Initialize array with zeros
    dFdz = ZeRo3D.copy()

    # Compute interior derivatives
    dFdz = dFdz.at[:, :, 1:nz].set(jnp.diff(F, axis=2) * idz)

    # Bottom boundary condition
    dFdz = dFdz.at[:, :, 0].set(0)

    return dFdz


# ============================================================
#  Vertical derivatives for a generic variable on w nodes
# ============================================================

@jax.jit
def Derivz_Generic_w(F, ZeRo3D):
    """
    Parameters:
    -----------
    F : ndarray of shape (nx, ny, nz)
        Generic variable defined on w nodes

    Returns:
    --------
    ndarray
        dFdz : Vertical derivative of F
    """

    # Initialize array with zeros
    dFdz = ZeRo3D.copy()

    # Compute interior derivatives
    dFdz = dFdz.at[:, :, 0:nz - 1].set(jnp.diff(F, axis=2) * idz)

    # Top boundary condition
    dFdz = dFdz.at[:, :, nz - 1].set(0)

    return dFdz
