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
File: NSE_AdvectionTerms.py
==================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: computes the advective terms as::

.. math::

    C_x = v\\left(\\frac{\\partial u}{\\partial y} - \\frac{\\partial v}{\\partial x}\\right) + w\\left(\\frac{\\partial u}{\\partial z} - \\frac{\\partial w}{\\partial x}\\right)

    C_y = u\\left(\\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y}\\right) + w\\left(\\frac{\\partial v}{\\partial z} - \\frac{\\partial w}{\\partial y}\\right)

    C_z = u\\left(\\frac{\\partial w}{\\partial x} - \\frac{\\partial u}{\\partial z}\\right) + v\\left(\\frac{\\partial w}{\\partial y} - \\frac{\\partial v}{\\partial z}\\right)
"""


# ============================================
# Imports
# ============================================

import jax
from ..config.Config import *
from ..config import Config

# Import derived variables
from ..config.DerivedVars import *

from ..utilities.Utilities import StagGridAvg

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealias functions
from ..operations.Dealiasing import Dealias1, Dealias2

# Import constants
from ..initialization.Preprocess import Constant
mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ------------------------------------------------------------------------------
# Use dealiasing in computing the advection terms
# ------------------------------------------------------------------------------

@jax.jit
def Advection_Dealias(u, v, w,
                      dudy, dudz,
                      dvdx, dvdz,
                      dwdx, dwdy,
                      ZeRo3D_fft, ZeRo3D_pad, ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    u, v, w : ndarray
        Velocity components in x, y, and z directions respectively.
        Shape: (nx, ny, nz)
    dudy, dudz : ndarray
        Derivatives of u-velocity with respect to y and z.
    dvdx, dvdz : ndarray
        Derivatives of v-velocity with respect to x and z.
    dwdx, dwdy : ndarray
        Derivatives of w-velocity with respect to x and y.

    Returns:
    --------
    tuple[ndarray, ndarray, ndarray]
        Cx : Convective term in x-direction
        Cy : Convective term in y-direction
        Cz : Convective term in z-direction
        Each array has shape (nx, ny, nz)
    """

    # Dealias
    u_pad = Dealias1(FFT(u), ZeRo3D_pad_fft)
    v_pad = Dealias1(FFT(v), ZeRo3D_pad_fft)
    w_pad = Dealias1(FFT(w), ZeRo3D_pad_fft)
    dudy_pad = Dealias1(FFT(dudy), ZeRo3D_pad_fft)
    dudz_pad = Dealias1(FFT(dudz), ZeRo3D_pad_fft)
    dvdx_pad = Dealias1(FFT(dvdx), ZeRo3D_pad_fft)
    dvdz_pad = Dealias1(FFT(dvdz), ZeRo3D_pad_fft)
    dwdx_pad = Dealias1(FFT(dwdx), ZeRo3D_pad_fft)
    dwdy_pad = Dealias1(FFT(dwdy), ZeRo3D_pad_fft)

    # Compute Cx: advection term in x-direction
    # First term: v*(∂u/∂y - ∂v/∂x)
    arg1 = v_pad * (dudy_pad - dvdx_pad)
    # Second term: w*(∂u/∂z - ∂w/∂x)
    arg2 = w_pad * (dudz_pad - dwdx_pad)

    # Apply staggered grid averaging to second term
    arg2G = StagGridAvg(arg2)

    # Combine terms with boundary condition handling
    cc = ZeRo3D_pad.copy()
    cc = cc.at[:, :, 1:nz-1].set(arg1[:, :, 1:nz-1] + arg2G[:, :, 1:nz-1])  # Interior points
    cc = cc.at[:, :, 0].set(arg1[:, :, 0] + 0.5 * arg2[:, :, 1])            # Bottom boundary
    cc = cc.at[:, :, nz-1].set(arg1[:, :, nz-1] + arg2[:, :, nz-1])         # Top boundary

    Cx = Dealias2(FFT_pad(cc), ZeRo3D_fft)

    # Compute Cy: advection term in y-direction
    # First term: u*(∂v/∂x - ∂u/∂y)
    arg1 = u_pad * (dvdx_pad - dudy_pad)
    # Second term: w*(∂v/∂z - ∂w/∂y)
    arg2 = w_pad * (dvdz_pad - dwdy_pad)

    # Apply staggered grid averaging to second term
    arg2G = StagGridAvg(arg2)

    cc = ZeRo3D_pad.copy()
    cc = cc.at[:, :, 1:nz-1].set(arg1[:, :, 1:nz-1] + arg2G[:, :, 1:nz-1])  # Interior points
    cc = cc.at[:, :, 0].set(arg1[:, :, 0] + 0.5 * arg2[:, :, 1])            # Bottom boundary
    cc = cc.at[:, :, nz-1].set(arg1[:, :, nz-1] + arg2[:, :, nz-1])         # Top boundary

    Cy = Dealias2(FFT_pad(cc), ZeRo3D_fft)

    # Compute Cz: advection term in z-direction
    arg1 = ZeRo3D_pad.copy()
    arg2 = ZeRo3D_pad.copy()

    # Average u and v components to staggered grid points
    u_pad_G = StagGridAvg(u_pad)
    v_pad_G = StagGridAvg(v_pad)

    # Compute interior points for both terms
    arg1 = arg1.at[:, :, 1:nz-1].set(u_pad_G[:, :, 0:nz-2] * (dwdx_pad[:, :, 1:nz-1] - dudz_pad[:, :, 1:nz-1]))
    arg2 = arg2.at[:, :, 1:nz-1].set(v_pad_G[:, :, 0:nz-2] * (dwdy_pad[:, :, 1:nz-1] - dvdz_pad[:, :, 1:nz-1]))

    # Combine terms and set boundary conditions
    cc = arg1 + arg2
    cc = cc.at[:, :, 0].set(0)     # Bottom boundary
    cc = cc.at[:, :, nz-1].set(0)  # Top boundary

    Cz = Dealias2(FFT_pad(cc), ZeRo3D_fft)

    return Cx, Cy, Cz


# ------------------------------------------------------------------------------
# Compute the advection terms without using any dealiasing
# ------------------------------------------------------------------------------

@jax.jit
def Advection_NoDealias(u, v, w,
                        dudy, dudz,
                        dvdx, dvdz,
                        dwdx, dwdy,
                        ZeRo3D):
    """
    Parameters:
    -----------
    u, v, w : ndarray, shape: (nx, ny, nz)
        Velocity components in x, y, and z directions respectively.
    dudy, dudz : ndarray
        Derivatives of u-velocity with respect to y and z.
    dvdx, dvdz : ndarray
        Derivatives of v-velocity with respect to x and z.
    dwdx, dwdy : ndarray
        Derivatives of w-velocity with respect to x and y.

    Returns:
    --------
    tuple[ndarray, ndarray, ndarray]
        Cx : advection term in x-direction
        Cy : advection term in y-direction
        Cz : advection term in z-direction
        Each array has shape (nx, ny, nz)

    Notes:
    ------
    - Function uses staggered grid averaging (StagGridAvg) for proper
      interpolation of velocities and derivatives
    """
    # Compute Cx: advection term in x-direction
    # First term: v*(∂u/∂y - ∂v/∂x)
    arg1 = v * (dudy - dvdx)
    # Second term: w*(∂u/∂z - ∂w/∂x)
    arg2 = w * (dudz - dwdx)

    # Apply staggered grid averaging to second term
    arg2G = StagGridAvg(arg2)

    # Combine terms with boundary condition handling
    Cx = ZeRo3D.copy()
    Cx = Cx.at[:, :, 1:nz-1].set(arg1[:, :, 1:nz-1] + arg2G[:, :, 1:nz-1])  # Interior points
    Cx = Cx.at[:, :, 0].set(arg1[:, :, 0] + 0.5 * arg2[:, :, 1])            # Bottom boundary
    Cx = Cx.at[:, :, nz-1].set(arg1[:, :, nz-1] + arg2[:, :, nz-1])         # Top boundary

    # Compute Cy: advection term in y-direction
    # First term: u*(∂v/∂x - ∂u/∂y)
    arg1 = u * (dvdx - dudy)
    # Second term: w*(∂v/∂z - ∂w/∂y)
    arg2 = w * (dvdz - dwdy)
    arg2G = StagGridAvg(arg2)

    Cy = ZeRo3D.copy()
    Cy = Cy.at[:, :, 1:nz-1].set(arg1[:, :, 1:nz-1] + arg2G[:, :, 1:nz-1])  # Interior points
    Cy = Cy.at[:, :, 0].set(arg1[:, :, 0] + 0.5 * arg2[:, :, 1])            # Bottom boundary
    Cy = Cy.at[:, :, nz-1].set(arg1[:, :, nz-1] + arg2[:, :, nz-1])         # Top boundary

    # Compute Cz: advection term in z-direction
    # Special handling required due to staggered grid arrangement
    arg1 = ZeRo3D.copy()
    arg2 = ZeRo3D.copy()

    # Average u and v components to staggered grid points
    u_G = StagGridAvg(u)
    v_G = StagGridAvg(v)

    # Compute interior points for both terms
    arg1 = arg1.at[:, :, 1:nz-1].set(u_G[:, :, 0:nz-2] * (dwdx[:, :, 1:nz-1] - dudz[:, :, 1:nz-1]))
    arg2 = arg2.at[:, :, 1:nz-1].set(v_G[:, :, 0:nz-2] * (dwdy[:, :, 1:nz-1] - dvdz[:, :, 1:nz-1]))

    # Combine terms and set boundary conditions
    Cz = arg1 + arg2
    Cz = Cz.at[:, :, 0].set(0)     # Bottom boundary
    Cz = Cz.at[:, :, nz-1].set(0)  # Top boundary

    return Cx, Cy, Cz


# ------------------------------------------------------------------------------
# Select one of the above functions based on dealias flag
# ------------------------------------------------------------------------------

@jax.jit
def Advection(
        u, v, w,
        dudy, dudz,
        dvdx, dvdz,
        dwdx, dwdy,
        ZeRo3D, ZeRo3D_fft,
        ZeRo3D_pad, ZeRo3D_pad_fft):

    if optDealias == 1:

        Cx, Cy, Cz = (
            Advection_Dealias(
                u, v, w,
                dudy, dudz,
                dvdx, dvdz,
                dwdx, dwdy,
                ZeRo3D_fft,
                ZeRo3D_pad, ZeRo3D_pad_fft))

    else:

        Cx, Cy, Cz = (
            Advection_NoDealias(
                u, v, w,
                dudy, dudz,
                dvdx, dvdz,
                dwdx, dwdy,
                ZeRo3D))

    return Cx, Cy, Cz
