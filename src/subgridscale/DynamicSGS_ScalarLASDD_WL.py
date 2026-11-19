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
File: DynamicSGS_ScalarLASDD_WL.py
===================================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-9
:Description: locally-averaged scale-dependent dynamic (LASDD) model
              for scalar transport using the Wong-Lilly (1994) SGS base
              model (LASDD-WL).
              Reference: Anderson, Basu, and Letchford (2007), EFM.
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT

# Import filtering functions
from ..operations.Filtering import Filtering_Level1, Filtering_Level2

# Import helper functions
from ..utilities.Utilities import PlanarMean, StagGridAvg
from ..utilities.Utilities import Imfilter


# ============================================================
# Find maximum real root between 0 and 5
# ============================================================

@jax.jit
def ComputeBeta2(a1, a3, a6, a8):
    """
    Scale-dependence parameter of the LASDD-WL scalar model, from the
    closed-form physical root of the ABL07 fifth-order polynomial.

    With x = α^(4/3)·β_θ, the polynomial of ABL07 Eq. A8/A10 factors exactly as
        P = a6·a8·(1 - x)^3·(1 + x)·[a1·a3/(a6·a8) - 1 - x],
    because the WL base model is linear in the strain (gradient). The triple
    root x = 1 makes M_ij (T_dn) vanish identically and x = -1 is negative, so
    the only physical root is
        β_θ = α^(-4/3)·(a1·a3/(a6·a8) - 1).
    Solving this directly avoids selecting the degenerate root (which the
    largest-root rule did whenever β_θ < α^(-4/3)) and the absolute-tolerance
    convergence test of the iterative solver.

    Parameters:
    -----------
    a1, a3, a6, a8 : ndarray
        1D arrays of the plane-averaged coefficients at each vertical level

    Returns:
    --------
    beta : ndarray
        1D array of β_θ at each level; 1.0 (scale invariance) where the root is
        not finite, outside (0, 5), or so close to α^(-4/3) that |1 - α^(4/3)·beta|
        is at round-off level (≤ √ε of the working precision). Near-singular but
        resolvable roots are kept; very large coefficients are handled downstream
        by the existing denominator and clipping guards.
    """
    # Exclude only the round-off-level singularity |1 - α^(4/3)·beta| ≈ 0 (M_ij = 0),
    # using √ε of the working precision (≈3.5e-4 float32, ≈1.5e-8 float64).
    BETA_DEG_TOL = jnp.sqrt(jnp.finfo(a1.dtype).eps)
    a68 = a6 * a8
    ratio = a1 * a3 / jnp.where(a68 == 0, jnp.nan, a68)
    beta = TFR ** (-4 / 3) * (ratio - 1)
    valid = (jnp.isfinite(beta) & (beta > 0) & (beta < 5.0) &
             (jnp.abs(1 - TFR ** (4 / 3) * beta) > BETA_DEG_TOL))
    return jnp.where(valid, beta, 1.0)


# ============================================================
# Compute CwlPrRatio coefficient at vertical level k
# ============================================================

@jax.jit
def CwlPrRatio_at_level_k(T_up_k, T_dn_k):
    """
    Parameters:
    -----------
    T_up_k : ndarray
        2D horizontal slice of T_up at level k
    T_dn_k : ndarray
        2D horizontal slice of T_dn at level k

    Returns:
    --------
    CwlPrRatio : ndarray
        2D array of C_WL/Pr_t at level k
    """

    T_up_F = Imfilter(T_up_k)
    T_dn_F = Imfilter(T_dn_k)

    CwlPrRatio = T_up_F / T_dn_F

    mask = (jnp.abs(T_dn_F) < 1e-10) | (CwlPrRatio < 0) | (CwlPrRatio > 1)
    CwlPrRatio = jnp.where(mask, 0.0, CwlPrRatio)

    return CwlPrRatio


# ============================================================
# Main LASDD-WL scalar code
# ============================================================

@jax.jit
def ScalarLASDD(
        u_, v_, w_,
        u_hat, v_hat, w_hat,
        u_hatd, v_hatd, w_hatd,
        TH,
        dTHdx, dTHdy, dTHdz,
        ZeRo3D):
    """
    Locally-averaged scale-dependent dynamic scalar model using the
    Wong-Lilly SGS base model (LASDD-WL).

    Parameters:
    -----------
    u_, v_, w_ : ndarray
        Interpolated velocity fields (from LASDD momentum step)
    u_hat, v_hat, w_hat : ndarray
        Level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : ndarray
        Level-2 filtered velocity components
    TH : ndarray
        Potential temperature field
    dTHdx, dTHdy, dTHdz : ndarray
        Potential temperature gradients
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    CwlPrRatio_3D : ndarray
        3D field of C_WL/Pr_t
    CwlPrRatio_1D : ndarray
        1D averaged profile of C_WL/Pr_t
    beta2_1D : ndarray
        1D profile of scalar scale-dependence parameter beta2
    """

    # TH is stored as anomaly (TH - T_0), so Leonard flux uTH_hat - u_hat*TH_hat
    # operates on small values (~0-5 K) with no catastrophic cancellation.
    TH_ = TH

    # Interpolate dTHdz to UVP nodes
    THz = ZeRo3D.copy()
    THz = THz.at[:, :, 1:nz - 1].set(StagGridAvg(dTHdz[:, :, 1:nz]))
    THz = THz.at[:, :, 0].set(dTHdz[:, :, 0])
    THz = THz.at[:, :, nz - 1].set(dTHdz[:, :, nz - 1])

    # Scalar flux products
    uTH = u_ * TH_
    vTH = v_ * TH_
    wTH = w_ * TH_

    # Level-1 filtered scalar and flux products
    TH_hat   = Filtering_Level1(FFT(TH_))
    uTH_hat  = Filtering_Level1(FFT(uTH))
    vTH_hat  = Filtering_Level1(FFT(vTH))
    wTH_hat  = Filtering_Level1(FFT(wTH))

    # Level-2 filtered scalar and flux products
    TH_hatd  = Filtering_Level2(FFT(TH_))
    uTH_hatd = Filtering_Level2(FFT(uTH))
    vTH_hatd = Filtering_Level2(FFT(vTH))
    wTH_hatd = Filtering_Level2(FFT(wTH))

    # Filtered scalar gradients (Level 1 and Level 2)
    dTHdx_hat  = Filtering_Level1(FFT(dTHdx))
    dTHdy_hat  = Filtering_Level1(FFT(dTHdy))
    dTHdz_hat  = Filtering_Level1(FFT(THz))

    dTHdx_hatd = Filtering_Level2(FFT(dTHdx))
    dTHdy_hatd = Filtering_Level2(FFT(dTHdy))
    dTHdz_hatd = Filtering_Level2(FFT(THz))

    # Scalar Leonard fluxes (ABL07 notation):
    #   K_i = LTH (Level 1), K'_i = QTH (Level 2)
    LTH11 = uTH_hat  - u_hat  * TH_hat
    LTH12 = vTH_hat  - v_hat  * TH_hat
    LTH13 = wTH_hat  - w_hat  * TH_hat

    QTH11 = uTH_hatd - u_hatd * TH_hatd
    QTH12 = vTH_hatd - v_hatd * TH_hatd
    QTH13 = wTH_hatd - w_hatd * TH_hatd

    # ----------------------------------------------------------
    # WL scalar polynomial coefficients (ABL07 Appendix)
    # Independent scalars: a1, a3, a6, a8
    # Level-1 flux LTH pairs with Level-1 gradients (dTH*_hat); Level-2 flux
    # QTH pairs with Level-2 gradients (dTH*_hatd), as in ABL07 Eq. A10 and
    # the original Fortran (optim_scl_wong.f, 2007).
    # ----------------------------------------------------------
    a1_terms = (QTH11 * dTHdx_hatd + QTH12 * dTHdy_hatd + QTH13 * dTHdz_hatd)
    a1 = PlanarMean(a1_terms)

    a3_terms = (dTHdx_hat ** 2 + dTHdy_hat ** 2 + dTHdz_hat ** 2)
    a3 = PlanarMean(a3_terms)

    a6_terms = (LTH11 * dTHdx_hat + LTH12 * dTHdy_hat + LTH13 * dTHdz_hat)
    a6 = PlanarMean(a6_terms)

    a8_terms = (dTHdx_hatd ** 2 + dTHdy_hatd ** 2 + dTHdz_hatd ** 2)
    a8 = PlanarMean(a8_terms)

    # The higher-order coefficients of ABL07 (a2, a4, a5, a7, a9, a10) are
    # multiples of a1, a3, a6 and a8; the fifth-order polynomial they form
    # factors exactly, and its physical root is computed in closed form.

    computeBeta = optSgs in [1, 2]
    if computeBeta:
        beta2_1D = ComputeBeta2(a1, a3, a6, a8)
    else:
        beta2_1D = jnp.ones(nz)
    beta2_3D = jnp.broadcast_to(beta2_1D.reshape(1, 1, nz), (nx, ny, nz))

    # ----------------------------------------------------------
    # WL scalar T_up and T_dn for CwlPrRatio (Level 1 only):
    #   M_i  = L^(4/3) * (1 - α^(4/3)*β2) * ∂_i c̄
    #   T_up = K_i M_i  = L^(4/3) * (1 - α^(4/3)*β2)   * K_i ∂_i c̄
    #   T_dn = M_i M_i  = L^(8/3) * (1 - α^(4/3)*β2)^2 * |∂_i c̄|²
    # with c̄ the Level-1 filtered scalar (cf. optim_scl_wong.f, 2007).
    # ----------------------------------------------------------
    factor = 1 - TFR ** (4 / 3) * beta2_3D

    T_up = L ** (4 / 3) * factor * a6_terms

    T_dn = L ** (8 / 3) * factor ** 2 * a3_terms

    CwlPrRatio_3D = jax.vmap(CwlPrRatio_at_level_k,
                              in_axes=(2, 2), out_axes=2)(T_up, T_dn)

    CwlPrRatio_1D = PlanarMean(CwlPrRatio_3D)

    return CwlPrRatio_3D, CwlPrRatio_1D, beta2_1D
