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
File: SGSStresses_STABSM.py
============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-22
:Description: SGS stress computation for the stability-dependent Smagorinsky
              model (STAB-SM, optSgs=5).

              Mixing-length formula (Mason-type wall limiter, non-dimensional):
                Lambda(z) = 1 / [1/(Cs*Delta)^2 + 1/(kappa*z_eff)^2]
                z_eff = z + z0m/z_scale

              Momentum stress:
                tau_ij = -2 * Lambda * fm(Ri) * |S| * S_ij

              Scalar flux (see ScalarSGSFluxes_STABSM.py):
                q_i = -Lambda * fh(Ri) * |S| * dTH/dxi   (no factor of 2)
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT, FFT_pad

# Import dealiasing functions
from ..operations.Dealiasing import Dealias1, Dealias2

# Import helper functions
from ..utilities.Utilities import StagGridAvg

# Reuse wall model from SM stresses (import only, no modification)
from .SGSStresses_SM import Wall


# ============================================================
# Build 1-D Lambda profiles  (z-only; broadcast to 3-D inside JIT)
# ============================================================
# All quantities are non-dimensional.
# UVP node k : z = (k + 0.5) * dz,   k = 0 .. nz-1
# W   node k : z = k * dz,            k = 0 .. nz-1
# z0m_nd = z0m / z_scale  (non-dim roughness length)
# Lambda = 1 / [1/(CsMO_SM*L)^2 + 1/(vonk*(z + z0m_nd))^2]

def _lambda_1d_uvp():
    z0m_nd = z0m / z_scale
    z_uvp  = (jnp.arange(nz) + 0.5) * dz
    lm     = vonk * (z_uvp + z0m_nd)
    return 1.0 / (1.0 / (CsMO_SM * L) ** 2 + 1.0 / lm ** 2)


def _lambda_1d_w():
    z0m_nd = z0m / z_scale
    z_w    = jnp.arange(nz) * dz
    # At k=0 (surface): lm = vonk * z0m_nd  (valid for z0m > 0)
    lm = vonk * (z_w + z0m_nd)
    return 1.0 / (1.0 / (CsMO_SM * L) ** 2 + 1.0 / lm ** 2)


# ============================================================
# Stability functions fm (momentum) and fh (heat)
# ============================================================

def _stability_fm_fh(Ri):
    """
    Element-wise stability correction (JAX: both branches always evaluated).

    Unstable  (Ri < 0):
        fm = sqrt(max(1 - cMO*Ri, 0))
        fh = aMO * sqrt(max(1 - bMO*Ri, 0))

    Stable sub-critical  (0 <= Ri < RicMO):
        fm = max(1 - Ri/Ric, 0)^r * (1 - hMO*Ri)
        fh = fMO * max(1 - Ri/Ric, 0)^r * (1 - gMO*Ri)

    Super-critical  (Ri >= RicMO):
        fm = fh = 0
    """
    fm_unstable = jnp.sqrt(jnp.maximum(1.0 - cMO_SM * Ri, 0.0))
    fh_unstable = aMO_SM * jnp.sqrt(jnp.maximum(1.0 - bMO_SM * Ri, 0.0))

    arg_stable = jnp.maximum(1.0 - Ri / RicMO_SM, 0.0)
    fm_stable  = (arg_stable ** rMO_SM) * (1.0 - hMO_SM * Ri)
    fh_stable  = fMO_SM * (arg_stable ** rMO_SM) * (1.0 - gMO_SM * Ri)

    fm = jnp.where(Ri < 0.0, fm_unstable,
         jnp.where(Ri < RicMO_SM, fm_stable, 0.0))
    fh = jnp.where(Ri < 0.0, fh_unstable,
         jnp.where(Ri < RicMO_SM, fh_stable, 0.0))
    return fm, fh


# ============================================================
# UVP-node stresses — with dealiasing
# ============================================================

@jax.jit
def StressesUVPnodes_Dealias_STABSM(
        S11_pad, S22_pad, S33_pad, S12_pad,
        S_uvp, S_uvp_pad,
        dTHdz,
        ZeRo3D,
        ZeRo3D_fft,
        ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    S11_pad .. S12_pad : ndarray
        Dealiased strain rate components at UVP nodes
    S_uvp : ndarray
        Strain rate magnitude at UVP nodes (non-padded, for Ri)
    S_uvp_pad : ndarray
        Dealiased strain rate magnitude (unused in stress; kept for interface
        consistency with other SGS modules)
    dTHdz : ndarray
        Vertical potential temperature gradient on W nodes
    ZeRo3D : ndarray
        Pre-allocated zero array (nx, ny, nz)
    ZeRo3D_fft : ndarray
        Pre-allocated FFT zero array for Dealias2
    ZeRo3D_pad_fft : ndarray
        Pre-allocated padded FFT zero array for Dealias1

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    Lambda_uvp2_3D : ndarray
        3-D Lambda field at UVP nodes (non-padded; passed to scalar function)
    fhS_uvp : ndarray
        fh * |S| at UVP nodes (non-padded; passed to scalar function)
    Lambda_uvp2_1D : ndarray (nz,)
        Horizontal mean of Lambda at UVP nodes (for statistics output)
    """

    # Lambda at UVP nodes
    Lambda_uvp2_1D  = _lambda_1d_uvp()
    Lambda_uvp2_3D  = jnp.broadcast_to(
        Lambda_uvp2_1D[None, None, :], (nx, ny, nz)).copy()
    Lambda_uvp2_pad = Dealias1(FFT(Lambda_uvp2_3D), ZeRo3D_pad_fft)

    # dTHdz lives on W nodes; average to UVP nodes for Richardson number
    THz_uvp = ZeRo3D.copy()
    THz_uvp = THz_uvp.at[:, :, 1:nz - 1].set(StagGridAvg(dTHdz[:, :, 1:nz]))
    THz_uvp = THz_uvp.at[:, :, 0].set(dTHdz[:, :, 0])
    THz_uvp = THz_uvp.at[:, :, nz - 1].set(dTHdz[:, :, nz - 1])

    S_safe = jnp.maximum(S_uvp, 1e-10)
    Ri_uvp = (g_nondim / T_0_nondim) * THz_uvp / S_safe ** 2

    fm_uvp, fh_uvp = _stability_fm_fh(Ri_uvp)

    fmS_uvp     = fm_uvp * S_uvp
    fhS_uvp     = fh_uvp * S_uvp          # returned for scalar reuse
    fmS_uvp_pad = Dealias1(FFT(fmS_uvp), ZeRo3D_pad_fft)

    # tau_ij = -2 * Lambda * fmS * S_ij
    preCompute = -2.0 * Lambda_uvp2_pad * fmS_uvp_pad
    txx_pad    = preCompute * S11_pad
    tyy_pad    = preCompute * S22_pad
    tzz_pad    = preCompute * S33_pad
    txy_pad    = preCompute * S12_pad

    txx_pad = txx_pad.at[:, :, nz - 1].set(0)
    tyy_pad = tyy_pad.at[:, :, nz - 1].set(0)
    tzz_pad = tzz_pad.at[:, :, nz - 1].set(0)
    txy_pad = txy_pad.at[:, :, nz - 1].set(0)

    txx = Dealias2(FFT_pad(txx_pad), ZeRo3D_fft)
    tyy = Dealias2(FFT_pad(tyy_pad), ZeRo3D_fft)
    tzz = Dealias2(FFT_pad(tzz_pad), ZeRo3D_fft)
    txy = Dealias2(FFT_pad(txy_pad), ZeRo3D_fft)

    return txx, tyy, tzz, txy, Lambda_uvp2_3D, fhS_uvp, Lambda_uvp2_1D


# ============================================================
# UVP-node stresses — without dealiasing
# ============================================================

@jax.jit
def StressesUVPnodes_NoDealias_STABSM(
        S11, S22, S33, S12,
        S_uvp,
        dTHdz,
        ZeRo3D):
    """
    Parameters:
    -----------
    S11, S22, S33, S12 : ndarray
        Strain rate components at UVP nodes
    S_uvp : ndarray
        Strain rate magnitude at UVP nodes
    dTHdz : ndarray
        Vertical potential temperature gradient on W nodes
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    txx, tyy, tzz, txy : ndarray
        SGS stress components at UVP nodes
    Lambda_uvp2_3D : ndarray
        3-D Lambda field at UVP nodes
    fhS_uvp : ndarray
        fh * |S| at UVP nodes
    Lambda_uvp2_1D : ndarray (nz,)
        Lambda profile (for statistics output)
    """

    Lambda_uvp2_1D = _lambda_1d_uvp()
    Lambda_uvp2_3D = jnp.broadcast_to(
        Lambda_uvp2_1D[None, None, :], (nx, ny, nz)).copy()

    THz_uvp = ZeRo3D.copy()
    THz_uvp = THz_uvp.at[:, :, 1:nz - 1].set(StagGridAvg(dTHdz[:, :, 1:nz]))
    THz_uvp = THz_uvp.at[:, :, 0].set(dTHdz[:, :, 0])
    THz_uvp = THz_uvp.at[:, :, nz - 1].set(dTHdz[:, :, nz - 1])

    S_safe = jnp.maximum(S_uvp, 1e-10)
    Ri_uvp = (g_nondim / T_0_nondim) * THz_uvp / S_safe ** 2

    fm_uvp, fh_uvp = _stability_fm_fh(Ri_uvp)

    fmS_uvp = fm_uvp * S_uvp
    fhS_uvp = fh_uvp * S_uvp

    preCompute = -2.0 * Lambda_uvp2_3D * fmS_uvp
    txx        = preCompute * S11
    tyy        = preCompute * S22
    tzz        = preCompute * S33
    txy        = preCompute * S12

    txx = txx.at[:, :, nz - 1].set(0)
    tyy = tyy.at[:, :, nz - 1].set(0)
    tzz = tzz.at[:, :, nz - 1].set(0)
    txy = txy.at[:, :, nz - 1].set(0)

    return txx, tyy, tzz, txy, Lambda_uvp2_3D, fhS_uvp, Lambda_uvp2_1D


# ============================================================
# W-node stresses — with dealiasing
# ============================================================

@jax.jit
def StressesWnodes_Dealias_STABSM(
        S13_pad, S23_pad,
        S_w, S_w_pad,
        dTHdz,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0,
        ZeRo3D_fft,
        ZeRo3D_pad_fft):
    """
    Parameters:
    -----------
    S13_pad, S23_pad : ndarray
        Dealiased shear strain components at W nodes
    S_w : ndarray
        Strain rate magnitude at W nodes (non-padded, for Richardson number)
    S_w_pad : ndarray
        Dealiased strain rate magnitude (shape reference for zero array)
    dTHdz : ndarray
        Vertical potential temperature gradient on W nodes
    u, v : ndarray
        Velocity components (for wall model)
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model
    ZeRo3D_fft : ndarray
        Pre-allocated FFT zero array for Dealias2
    ZeRo3D_pad_fft : ndarray
        Pre-allocated padded FFT zero array for Dealias1

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    Lambda_w2_3D : ndarray
        3-D Lambda field at W nodes (non-padded; passed to scalar function)
    fhS_w : ndarray
        fh * |S| at W nodes (non-padded; passed to scalar function)
    """

    Lambda_w2_1D  = _lambda_1d_w()
    Lambda_w2_3D  = jnp.broadcast_to(
        Lambda_w2_1D[None, None, :], (nx, ny, nz)).copy()
    Lambda_w2_pad = Dealias1(FFT(Lambda_w2_3D), ZeRo3D_pad_fft)

    # dTHdz is already on W nodes
    S_safe_w = jnp.maximum(S_w, 1e-10)
    Ri_w     = (g_nondim / T_0_nondim) * dTHdz / S_safe_w ** 2

    fm_w, fh_w = _stability_fm_fh(Ri_w)

    fmS_w     = fm_w * S_w
    fhS_w     = fh_w * S_w         # returned for scalar reuse
    fmS_w_pad = Dealias1(FFT(fmS_w), ZeRo3D_pad_fft)

    # Lambda_w is defined directly at W nodes — no StagGridAvg needed
    txz_pad = jnp.zeros_like(S_w_pad)
    tyz_pad = jnp.zeros_like(S_w_pad)

    preCompute = -2.0 * Lambda_w2_pad[:, :, 1:nz - 1] * fmS_w_pad[:, :, 1:nz - 1]
    txz_pad = txz_pad.at[:, :, 1:nz - 1].set(preCompute * S13_pad[:, :, 1:nz - 1])
    tyz_pad = tyz_pad.at[:, :, 1:nz - 1].set(preCompute * S23_pad[:, :, 1:nz - 1])

    txz_pad = txz_pad.at[:, :, nz - 1].set(0)
    tyz_pad = tyz_pad.at[:, :, nz - 1].set(0)

    txz = Dealias2(FFT_pad(txz_pad), ZeRo3D_fft)
    tyz = Dealias2(FFT_pad(tyz_pad), ZeRo3D_fft)

    txz_wall, tyz_wall = Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0)
    txz = txz.at[:, :, 0].set(txz_wall)
    tyz = tyz.at[:, :, 0].set(tyz_wall)

    return txz, tyz, Lambda_w2_3D, fhS_w


# ============================================================
# W-node stresses — without dealiasing
# ============================================================

@jax.jit
def StressesWnodes_NoDealias_STABSM(
        S13, S23,
        S_w,
        dTHdz,
        u, v, M_sfc_loc, psi2D_m, psi2D_m0):
    """
    Parameters:
    -----------
    S13, S23 : ndarray
        Shear strain components at W nodes
    S_w : ndarray
        Strain rate magnitude at W nodes
    dTHdz : ndarray
        Vertical potential temperature gradient on W nodes
    u, v : ndarray
        Velocity components (for wall model)
    M_sfc_loc, psi2D_m, psi2D_m0 : ndarray
        Surface parameters for wall model

    Returns:
    --------
    txz, tyz : ndarray
        SGS stress components at W nodes
    Lambda_w2_3D : ndarray
        3-D Lambda field at W nodes
    fhS_w : ndarray
        fh * |S| at W nodes
    """

    Lambda_w2_1D = _lambda_1d_w()
    Lambda_w2_3D = jnp.broadcast_to(
        Lambda_w2_1D[None, None, :], (nx, ny, nz)).copy()

    S_safe_w = jnp.maximum(S_w, 1e-10)
    Ri_w     = (g_nondim / T_0_nondim) * dTHdz / S_safe_w ** 2

    fm_w, fh_w = _stability_fm_fh(Ri_w)

    fmS_w = fm_w * S_w
    fhS_w = fh_w * S_w

    txz = jnp.zeros_like(S_w)
    tyz = jnp.zeros_like(S_w)

    preCompute = -2.0 * Lambda_w2_3D[:, :, 1:nz - 1] * fmS_w[:, :, 1:nz - 1]
    txz = txz.at[:, :, 1:nz - 1].set(preCompute * S13[:, :, 1:nz - 1])
    tyz = tyz.at[:, :, 1:nz - 1].set(preCompute * S23[:, :, 1:nz - 1])

    txz = txz.at[:, :, nz - 1].set(0)
    tyz = tyz.at[:, :, nz - 1].set(0)

    txz_wall, tyz_wall = Wall(u, v, M_sfc_loc, psi2D_m, psi2D_m0)
    txz = txz.at[:, :, 0].set(txz_wall)
    tyz = tyz.at[:, :, 0].set(tyz_wall)

    return txz, tyz, Lambda_w2_3D, fhS_w
