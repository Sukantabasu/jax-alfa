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
File: DynamicSGS_LASDD_WL.py
=============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2026-5-9
:Description: locally-averaged scale-dependent dynamic (LASDD) model
              using the Wong-Lilly (1994) SGS base model (LASDD-WL).
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
def ComputeBeta1(a1, a3, a6, a8):
    """
    Scale-dependence parameter of the LASDD-WL momentum model, from the
    closed-form physical root of the ABL07 fifth-order polynomial.

    With x = α^(4/3)·β, the polynomial of ABL07 Eq. A8/A10 factors exactly as
        P = a6·a8·(1 - x)^3·(1 + x)·[a1·a3/(a6·a8) - 1 - x],
    because the WL base model is linear in the strain (gradient). The triple
    root x = 1 makes M_ij (T_dn) vanish identically and x = -1 is negative, so
    the only physical root is
        β = α^(-4/3)·(a1·a3/(a6·a8) - 1).
    Solving this directly avoids selecting the degenerate root (which the
    largest-root rule did whenever β < α^(-4/3)) and the absolute-tolerance
    convergence test of the iterative solver.

    Parameters:
    -----------
    a1, a3, a6, a8 : ndarray
        1D arrays of the plane-averaged coefficients at each vertical level

    Returns:
    --------
    beta : ndarray
        1D array of β at each level; 1.0 (scale invariance) where the root is
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
# Compute Wong-Lilly coefficient at vertical level k
# ============================================================

@jax.jit
def Cwl_at_level_k(LM_k, MM_k):
    """
    Parameters:
    -----------
    LM_k : jax.numpy.ndarray
        2D horizontal slice of LM at level k
    MM_k : jax.numpy.ndarray
        2D horizontal slice of MM at level k

    Returns:
    --------
    Cwl : ndarray
        2D array of the Wong-Lilly coefficient C_WL at level k
    """

    LMx = Imfilter(LM_k)
    MMx = Imfilter(MM_k)

    Cwl = LMx / MMx

    mask = (jnp.abs(MMx) < 1e-10) | (Cwl < 0) | (Cwl > 1)
    Cwl = jnp.where(mask, 0.0, Cwl)

    return Cwl


# ============================================================
# Main LASDD-WL code
# ============================================================

@jax.jit
def LASDD(
        u, v, w,
        S11, S22, S33,
        S12, S13, S23,
        ZeRo3D):
    """
    Locally-averaged scale-dependent dynamic model using the
    Wong-Lilly SGS base model (LASDD-WL).

    Parameters:
    -----------
    u, v, w : ndarray
        Velocity components
    S11, S22, S33 : ndarray
        Normal strain rate components
    S12, S13, S23 : ndarray
        Shear strain rate components
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    u_, v_, w_ : ndarray
        Interpolated velocity components
    u_hat, v_hat, w_hat : ndarray
        Level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : ndarray
        Level-2 filtered velocity components
    S_hat, S_hatd : ndarray
        Filtered strain rate magnitudes (diagnostic)
    Cwl_3D : ndarray
        3D field of C_WL coefficient
    Cwl_1D_avg1 : ndarray
        1D profile of C_WL (square of mean of sqrt)
    Cwl_1D_avg2 : ndarray
        1D profile of C_WL (direct mean)
    beta1_1D : ndarray
        1D profile of scale-dependence parameter beta1
    """

    u_ = u.copy()
    v_ = v.copy()
    w_ = ZeRo3D.copy()
    w_ = w_.at[:, :, 1:nz - 1].set(StagGridAvg(w[:, :, 1:nz]))
    w_ = w_.at[:, :, 0].set(0.5 * w[:, :, 1])
    w_ = w_.at[:, :, nz - 1].set(w[:, :, nz - 2])

    # Velocity products
    uu, vv, ww = u_ ** 2, v_ ** 2, w_ ** 2
    uv, uw, vw = u_ * v_, u_ * w_, v_ * w_

    # Level-1 filtered velocities and products
    u_hat  = Filtering_Level1(FFT(u_))
    v_hat  = Filtering_Level1(FFT(v_))
    w_hat  = Filtering_Level1(FFT(w_))
    uu_hat = Filtering_Level1(FFT(uu))
    vv_hat = Filtering_Level1(FFT(vv))
    ww_hat = Filtering_Level1(FFT(ww))
    uv_hat = Filtering_Level1(FFT(uv))
    uw_hat = Filtering_Level1(FFT(uw))
    vw_hat = Filtering_Level1(FFT(vw))

    # Level-2 filtered velocities and products
    u_hatd  = Filtering_Level2(FFT(u_))
    v_hatd  = Filtering_Level2(FFT(v_))
    w_hatd  = Filtering_Level2(FFT(w_))
    uu_hatd = Filtering_Level2(FFT(uu))
    vv_hatd = Filtering_Level2(FFT(vv))
    ww_hatd = Filtering_Level2(FFT(ww))
    uv_hatd = Filtering_Level2(FFT(uv))
    uw_hatd = Filtering_Level2(FFT(uw))
    vw_hatd = Filtering_Level2(FFT(vw))

    # Filtered strain rate components
    S11_hat  = Filtering_Level1(FFT(S11))
    S22_hat  = Filtering_Level1(FFT(S22))
    S33_hat  = Filtering_Level1(FFT(S33))
    S12_hat  = Filtering_Level1(FFT(S12))
    S13_hat  = Filtering_Level1(FFT(S13))
    S23_hat  = Filtering_Level1(FFT(S23))

    S11_hatd = Filtering_Level2(FFT(S11))
    S22_hatd = Filtering_Level2(FFT(S22))
    S33_hatd = Filtering_Level2(FFT(S33))
    S12_hatd = Filtering_Level2(FFT(S12))
    S13_hatd = Filtering_Level2(FFT(S13))
    S23_hatd = Filtering_Level2(FFT(S23))

    # Filtered strain rate magnitudes (diagnostic outputs)
    S_hat  = jnp.sqrt(2 * (S11_hat ** 2 + S22_hat ** 2 + S33_hat ** 2 +
                           2 * S12_hat ** 2 +
                           2 * S13_hat ** 2 +
                           2 * S23_hat ** 2))
    S_hatd = jnp.sqrt(2 * (S11_hatd ** 2 + S22_hatd ** 2 + S33_hatd ** 2 +
                            2 * S12_hatd ** 2 +
                            2 * S13_hatd ** 2 +
                            2 * S23_hatd ** 2))

    # Leonard stress tensors L_ij (Level 1) and Q_ij (Level 2)
    L11, L22, L33 = (uu_hat - u_hat ** 2,
                     vv_hat - v_hat ** 2,
                     ww_hat - w_hat ** 2)
    L12, L13, L23 = (uv_hat - u_hat * v_hat,
                     uw_hat - u_hat * w_hat,
                     vw_hat - v_hat * w_hat)

    Q11, Q22, Q33 = (uu_hatd - u_hatd ** 2,
                     vv_hatd - v_hatd ** 2,
                     ww_hatd - w_hatd ** 2)
    Q12, Q13, Q23 = (uv_hatd - u_hatd * v_hatd,
                     uw_hatd - u_hatd * w_hatd,
                     vw_hatd - v_hatd * w_hatd)

    # ----------------------------------------------------------
    # WL polynomial coefficients (ABL07 Appendix, Eqs. A1-A10)
    # Independent scalars: a1, a3, a6, a8
    # Level-1 Leonard stress L_ij pairs with Level-1 strain (S_ij_hat);
    # Level-2 Leonard stress Q_ij pairs with Level-2 strain (S_ij_hatd),
    # as in ABL07 Eq. A8 and the original Fortran (optim_wong.f, 2007).
    # ----------------------------------------------------------
    a1_terms = (Q11 * S11_hatd + Q22 * S22_hatd + Q33 * S33_hatd +
                2 * (Q12 * S12_hatd + Q13 * S13_hatd + Q23 * S23_hatd))
    a1 = PlanarMean(a1_terms)

    a3_terms = (S11_hat ** 2 + S22_hat ** 2 + S33_hat ** 2 +
                2 * (S12_hat ** 2 + S13_hat ** 2 + S23_hat ** 2))
    a3 = PlanarMean(a3_terms)

    a6_terms = (L11 * S11_hat + L22 * S22_hat + L33 * S33_hat +
                2 * (L12 * S12_hat + L13 * S13_hat + L23 * S23_hat))
    a6 = PlanarMean(a6_terms)

    a8_terms = (S11_hatd ** 2 + S22_hatd ** 2 + S33_hatd ** 2 +
                2 * (S12_hatd ** 2 + S13_hatd ** 2 + S23_hatd ** 2))
    a8 = PlanarMean(a8_terms)

    # The higher-order coefficients of ABL07 (a2, a4, a5, a7, a9, a10) are
    # multiples of a1, a3, a6 and a8; the fifth-order polynomial they form
    # factors exactly, and its physical root is computed in closed form.

    computeBeta = optSgs in [1, 2]
    if computeBeta:
        beta1_1D = ComputeBeta1(a1, a3, a6, a8)
    else:
        beta1_1D = jnp.ones(nz)
    beta1_3D = jnp.broadcast_to(beta1_1D.reshape(1, 1, nz), (nx, ny, nz))

    # ----------------------------------------------------------
    # WL M tensor (ABL07 Eq. A3): M_ij = 2*Δf^(4/3)*(1 - α^(4/3)*β)*S̄_ij,
    # with the Level-1 strain S̄_ij (S_ij_hat) in both terms. The WL base
    # model is linear in S, so the test-filtered SGS stress is
    # proportional to S̄_ij itself (cf. optim_wong.f, 2007).
    # ----------------------------------------------------------
    T1 = 2 * L ** (4 / 3)
    T2 = 2 * (TFR * L) ** (4 / 3)
    M11 = T1 * S11_hat - T2 * beta1_3D * S11_hat
    M22 = T1 * S22_hat - T2 * beta1_3D * S22_hat
    M33 = T1 * S33_hat - T2 * beta1_3D * S33_hat
    M12 = T1 * S12_hat - T2 * beta1_3D * S12_hat
    M13 = T1 * S13_hat - T2 * beta1_3D * S13_hat
    M23 = T1 * S23_hat - T2 * beta1_3D * S23_hat

    # LM = L_ij * M_ij,  MM = M_ij * M_ij
    LM = ((L11 * M11 + L22 * M22 + L33 * M33) +
          2 * (L12 * M12 + L13 * M13 + L23 * M23))

    MM = (M11 ** 2 + M22 ** 2 + M33 ** 2 +
          2 * (M12 ** 2 + M13 ** 2 + M23 ** 2))

    # C_WL field: local 3x3 averaging via Imfilter
    Cwl_3D = jax.vmap(Cwl_at_level_k, in_axes=(2, 2), out_axes=2)(LM, MM)

    Cwl_1D_avg1 = PlanarMean(jnp.sqrt(Cwl_3D)) ** 2
    Cwl_1D_avg2 = PlanarMean(Cwl_3D)

    return (u_, v_, w_,
            u_hat, v_hat, w_hat,
            u_hatd, v_hatd, w_hatd,
            S_hat, S_hatd,
            Cwl_3D, Cwl_1D_avg1, Cwl_1D_avg2, beta1_1D)
