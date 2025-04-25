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
File: SGS_StrainRates.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-7
:Description: computes strain rate tensors and their magnitudes
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *
from ..config import Config

# Import derived variables
from ..config.DerivedVars import *

# Import helper functions
from ..utilities.Utilities import StagGridAvg

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealiasing modules
from ..operations.Dealiasing import Dealias1


# ============================================================
# Compute strain rates on uvp nodes with dealiasing
# ============================================================

@jax.jit
def StrainsUVPnodes_Dealias(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        ZeRo3D, ZeRo3D_pad_fft):
    """
    Computes strain rate tensors at UVP nodes with dealiasing.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray
        Derivatives of velocity components in z-direction
    ZeRo3D : ndarray
        Pre-allocated zero array
    ZeRo3D_pad_fft : ndarray
        Pre-allocated array for dealiasing

    Returns:
    --------
    S11, S22, S33, S12, S13, S23 : ndarray
        Strain rate tensor components
    S : ndarray
        Strain rate magnitude
    S11_pad, S22_pad, S33_pad, S12_pad, S13_pad, S23_pad : ndarray
        Dealiased strain rate components
    S_pad : ndarray
        Dealiased strain rate magnitude
    """

    # Initialize arrays
    uz, vz = ZeRo3D.copy(), ZeRo3D.copy()
    wx, wy = ZeRo3D.copy(), ZeRo3D.copy()

    # Average derivatives to uvp nodes
    uz = uz.at[:, :, 1:nz - 1].set(StagGridAvg(dudz[:, :, 1:nz]))
    vz = vz.at[:, :, 1:nz - 1].set(StagGridAvg(dvdz[:, :, 1:nz]))
    wx = wx.at[:, :, 1:nz - 1].set(StagGridAvg(dwdx[:, :, 1:nz]))
    wy = wy.at[:, :, 1:nz - 1].set(StagGridAvg(dwdy[:, :, 1:nz]))

    # Set boundary conditions
    uz = uz.at[:, :, 0].set(dudz[:, :, 0])
    vz = vz.at[:, :, 0].set(dvdz[:, :, 0])
    wx = wx.at[:, :, 0].set(0.5 * dwdx[:, :, 1])
    wy = wy.at[:, :, 0].set(0.5 * dwdy[:, :, 1])
    uz = uz.at[:, :, nz - 1].set(dudz[:, :, nz - 1])
    vz = vz.at[:, :, nz - 1].set(dvdz[:, :, nz - 1])
    wx = wx.at[:, :, nz - 1].set(dwdx[:, :, nz - 1])
    wy = wy.at[:, :, nz - 1].set(dwdy[:, :, nz - 1])

    # Compute strain rate tensors on uvp nodes
    S11, S22, S33 = dudx.copy(), dvdy.copy(), dwdz.copy()
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)

    # Compute strain rate magnitude
    S = jnp.sqrt(2 * (S11 ** 2 + S22 ** 2 + S33 ** 2 +
                      2 * S12 ** 2 + 2 * S13 ** 2 + 2 * S23 ** 2))

    # Dealiased variables 
    S11_pad = Dealias1(FFT(S11), ZeRo3D_pad_fft)
    S22_pad = Dealias1(FFT(S22), ZeRo3D_pad_fft)
    S33_pad = Dealias1(FFT(S33), ZeRo3D_pad_fft)
    S12_pad = Dealias1(FFT(S12), ZeRo3D_pad_fft)
    S13_pad = Dealias1(FFT(S13), ZeRo3D_pad_fft)
    S23_pad = Dealias1(FFT(S23), ZeRo3D_pad_fft)

    # Compute dealiased strain rate magnitude
    S_pad = jnp.sqrt(2 * (S11_pad ** 2 + S22_pad ** 2 + S33_pad ** 2 +
                          2 * S12_pad ** 2 + 2 * S13_pad ** 2 + 2 * S23_pad ** 2))

    return (
        S11, S22, S33,
        S12, S13, S23,
        S,
        S11_pad, S22_pad, S33_pad,
        S12_pad, S13_pad, S23_pad,
        S_pad)


# ====================================================
# Compute strain rates on uvp nodes without dealiasing
# ====================================================

@jax.jit
def StrainsUVPnodes_NoDealias(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        ZeRo3D):
    """
    Computes strain rate tensors at UVP nodes without dealiasing.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray
        Derivatives of velocity components in z-direction
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    S11, S22, S33, S12, S13, S23 : ndarray
        Strain rate tensor components
    S : ndarray
        Strain rate magnitude
    """

    # Initialize arrays
    uz, vz = ZeRo3D.copy(), ZeRo3D.copy()
    wx, wy = ZeRo3D.copy(), ZeRo3D.copy()

    # Average derivatives to uvp nodes
    uz = uz.at[:, :, 1:nz - 1].set(StagGridAvg(dudz[:, :, 1:nz]))
    vz = vz.at[:, :, 1:nz - 1].set(StagGridAvg(dvdz[:, :, 1:nz]))
    wx = wx.at[:, :, 1:nz - 1].set(StagGridAvg(dwdx[:, :, 1:nz]))
    wy = wy.at[:, :, 1:nz - 1].set(StagGridAvg(dwdy[:, :, 1:nz]))

    # Set boundary conditions
    uz = uz.at[:, :, 0].set(dudz[:, :, 0])
    vz = vz.at[:, :, 0].set(dvdz[:, :, 0])
    wx = wx.at[:, :, 0].set(0.5 * dwdx[:, :, 1])
    wy = wy.at[:, :, 0].set(0.5 * dwdy[:, :, 1])
    uz = uz.at[:, :, nz - 1].set(dudz[:, :, nz - 1])
    vz = vz.at[:, :, nz - 1].set(dvdz[:, :, nz - 1])
    wx = wx.at[:, :, nz - 1].set(dwdx[:, :, nz - 1])
    wy = wy.at[:, :, nz - 1].set(dwdy[:, :, nz - 1])

    # Compute strain rate tensors on uvp nodes
    S11, S22, S33 = dudx.copy(), dvdy.copy(), dwdz.copy()
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)

    # Compute strain rate magnitude
    S = jnp.sqrt(2 * (S11 ** 2 + S22 ** 2 + S33 ** 2 +
                      2 * S12 ** 2 + 2 * S13 ** 2 + 2 * S23 ** 2))

    return (
        S11, S22, S33,
        S12, S13, S23,
        S)


# =================================================
# Compute strain rates on w nodes with dealiasing
# =================================================

@jax.jit
def StrainsWnodes_Dealias(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        ZeRo3D, ZeRo3D_pad_fft):
    """
    Computes strain rate tensors at W nodes with dealiasing.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray
        Derivatives of velocity components in z-direction
    ZeRo3D : ndarray
        Pre-allocated zero array
    ZeRo3D_pad_fft : ndarray
        Pre-allocated array for dealiasing

    Returns:
    --------
    S13_pad, S23_pad : ndarray
        Dealiased shear strain components at W nodes
    S_pad : ndarray
        Dealiased strain rate magnitude at W nodes
    """

    # Compute stresses at w-levels
    ux, uy = ZeRo3D.copy(), ZeRo3D.copy()
    vx, vy = ZeRo3D.copy(), ZeRo3D.copy()
    wz = ZeRo3D.copy()

    # Average to w-levels
    ux = ux.at[:, :, 1:nz-1].set(StagGridAvg(dudx[:, :, :nz - 1]))
    uy = uy.at[:, :, 1:nz-1].set(StagGridAvg(dudy[:, :, :nz - 1]))
    vx = vx.at[:, :, 1:nz-1].set(StagGridAvg(dvdx[:, :, :nz - 1]))
    vy = vy.at[:, :, 1:nz-1].set(StagGridAvg(dvdy[:, :, :nz - 1]))
    uz, vz = dudz.copy(), dvdz.copy()
    wx, wy = dwdx.copy(), dwdy.copy()
    wz = wz.at[:, :, 1:nz-1].set(StagGridAvg(dwdz[:, :, :nz - 1]))

    # Set bottom boundary conditions
    ux = ux.at[:, :, 0].set(dudx[:, :, 0])
    uy = uy.at[:, :, 0].set(dudy[:, :, 0])
    vx = vx.at[:, :, 0].set(dvdx[:, :, 0])
    vy = vy.at[:, :, 0].set(dvdy[:, :, 0])
    wx = wx.at[:, :, 0].set(0.5 * (dwdx[:, :, 0] + dwdx[:, :, 1]))
    wy = wy.at[:, :, 0].set(0.5 * (dwdy[:, :, 0] + dwdy[:, :, 1]))
    wz = wz.at[:, :, 0].set(dwdz[:, :, 0])

    # Compute strain rates at w-levels
    S11, S22, S33 = ux.copy(), vy.copy(), wz.copy()
    S12 = 0.5 * (uy + vx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)

    # These variables are defined on w nodes
    S11_pad = Dealias1(FFT(S11), ZeRo3D_pad_fft)
    S22_pad = Dealias1(FFT(S22), ZeRo3D_pad_fft)
    S33_pad = Dealias1(FFT(S33), ZeRo3D_pad_fft)
    S12_pad = Dealias1(FFT(S12), ZeRo3D_pad_fft)
    S13_pad = Dealias1(FFT(S13), ZeRo3D_pad_fft)
    S23_pad = Dealias1(FFT(S23), ZeRo3D_pad_fft)

    # Compute strain rate magnitude at w-levels
    S_pad = jnp.sqrt(2 * (S11_pad ** 2 + S22_pad ** 2 + S33_pad ** 2 +
                          2 * S12_pad ** 2 + 2 * S13_pad ** 2 + 2 * S23_pad ** 2))

    return (S13_pad, S23_pad,
            S_pad)


# ====================================================
# Compute strain rates on uvp nodes without dealiasing
# ====================================================

@jax.jit
def StrainsWnodes_NoDealias(
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        ZeRo3D):
    """
    Computes strain rate tensors at W nodes without dealiasing.

    Parameters:
    -----------
    dudx, dvdx, dwdx : ndarray
        Derivatives of velocity components in x-direction
    dudy, dvdy, dwdy : ndarray
        Derivatives of velocity components in y-direction
    dudz, dvdz, dwdz : ndarray
        Derivatives of velocity components in z-direction
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    S13, S23 : ndarray
        Shear strain components at W nodes
    S : ndarray
        Strain rate magnitude at W nodes
    """

    # Compute stresses at w-levels
    ux, uy = ZeRo3D.copy(), ZeRo3D.copy()
    vx, vy = ZeRo3D.copy(), ZeRo3D.copy()
    wz = ZeRo3D.copy()

    # Average to w-levels
    ux = ux.at[:, :, 1:nz-1].set(StagGridAvg(dudx[:, :, :nz - 1]))
    uy = uy.at[:, :, 1:nz-1].set(StagGridAvg(dudy[:, :, :nz - 1]))
    vx = vx.at[:, :, 1:nz-1].set(StagGridAvg(dvdx[:, :, :nz - 1]))
    vy = vy.at[:, :, 1:nz-1].set(StagGridAvg(dvdy[:, :, :nz - 1]))
    uz, vz = dudz.copy(), dvdz.copy()
    wx, wy = dwdx.copy(), dwdy.copy()
    wz = wz.at[:, :, 1:nz-1].set(StagGridAvg(dwdz[:, :, :nz - 1]))

    # Set bottom boundary conditions
    ux = ux.at[:, :, 0].set(dudx[:, :, 0])
    uy = uy.at[:, :, 0].set(dudy[:, :, 0])
    vx = vx.at[:, :, 0].set(dvdx[:, :, 0])
    vy = vy.at[:, :, 0].set(dvdy[:, :, 0])
    wx = wx.at[:, :, 0].set(0.5 * (dwdx[:, :, 0] + dwdx[:, :, 1]))
    wy = wy.at[:, :, 0].set(0.5 * (dwdy[:, :, 0] + dwdy[:, :, 1]))
    wz = wz.at[:, :, 0].set(dwdz[:, :, 0])

    # Compute strain rates at w-levels
    S11, S22, S33 = ux.copy(), vy.copy(), wz.copy()
    S12 = 0.5 * (uy + vx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)

    # Compute strain rate magnitude at w-levels
    S = jnp.sqrt(2 * (S11 ** 2 + S22 ** 2 + S33 ** 2 +
                      2 * S12 ** 2 + 2 * S13 ** 2 + 2 * S23 ** 2))

    return (S13, S23,
            S)
