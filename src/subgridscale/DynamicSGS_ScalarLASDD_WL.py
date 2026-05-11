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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
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
from ..utilities.Utilities import Roots, Imfilter


# ============================================================
# Find maximum real root between 0 and 5
# ============================================================

@jax.jit
def ComputeBeta2(ff, ee, dd, cc, bb, aa):
    """
    Solves the polynomial:
    ff*x^5 + ee*x^4 + dd*x^3 + cc*x^2 + bb*x + aa = 0
    for each vertical level to find the scalar scale-dependence
    parameter beta2 used in the LASDD-WL scalar model.

    Parameters:
    -----------
    ff, ee, dd, cc, bb, aa : ndarray
        1D arrays of polynomial coefficients at each vertical level

    Returns:
    --------
    beta2 : ndarray
        1D array of the maximum valid real root for each vertical level
    """

    def find_roots_for_level(k):
        coeffs = jnp.array([ff[k], ee[k], dd[k], cc[k], bb[k], aa[k]])
        guesses = jnp.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5])
        roots = jax.vmap(lambda guess:
                         Roots(coeffs, init_guess=guess))(guesses)

        valid_roots = jnp.where(
            (jnp.abs(jnp.imag(roots)) < 1e-10) &
            (jnp.real(roots) > 0) &
            (jnp.real(roots) < 5.0),
            jnp.real(roots),
            jnp.nan
        )

        max_root = jnp.nanmax(valid_roots)
        return jnp.where(jnp.isnan(max_root), 1.0, max_root)

    return jax.vmap(find_roots_for_level)(jnp.arange(ff.shape[0]))


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

    TH_ = TH.copy()

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
    TH_hat   = Filtering_Level1(FFT(TH))
    uTH_hat  = Filtering_Level1(FFT(uTH))
    vTH_hat  = Filtering_Level1(FFT(vTH))
    wTH_hat  = Filtering_Level1(FFT(wTH))

    # Level-2 filtered scalar and flux products
    TH_hatd  = Filtering_Level2(FFT(TH))
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

    # Scalar Leonard fluxes:
    #   K'_i = LTH (Level 1), K_i = QTH (Level 2)
    LTH11 = uTH_hat  - u_hat  * TH_hat
    LTH12 = vTH_hat  - v_hat  * TH_hat
    LTH13 = wTH_hat  - w_hat  * TH_hat

    QTH11 = uTH_hatd - u_hatd * TH_hatd
    QTH12 = vTH_hatd - v_hatd * TH_hatd
    QTH13 = wTH_hatd - w_hatd * TH_hatd

    # ----------------------------------------------------------
    # WL scalar polynomial coefficients (ABL07 Appendix)
    # Independent scalars: a1, a3, a6, a8
    # ----------------------------------------------------------
    a1_terms = (QTH11 * dTHdx_hat + QTH12 * dTHdy_hat + QTH13 * dTHdz_hat)
    a1 = PlanarMean(a1_terms)

    a3_terms = (dTHdx_hat ** 2 + dTHdy_hat ** 2 + dTHdz_hat ** 2)
    a3 = PlanarMean(a3_terms)

    a6_terms = (LTH11 * dTHdx_hat + LTH12 * dTHdy_hat + LTH13 * dTHdz_hat)
    a6 = PlanarMean(a6_terms)

    a8_terms = (dTHdx_hatd ** 2 + dTHdy_hatd ** 2 + dTHdz_hatd ** 2)
    a8 = PlanarMean(a8_terms)

    # Derived scalars
    a2  = -(TFR ** (8 / 3)) * a1
    a4  = -2 * TFR ** (4 / 3) * a3
    a5  =  TFR ** (8 / 3) * a3
    a7  = -TFR ** (4 / 3) * a6
    a9  = -2 * TFR ** (8 / 3) * a8
    a10 =  TFR ** (16 / 3) * a8

    # Polynomial coefficients A0...A5 mapped to aa...ff
    aa = a1 * a3 - a6 * a8            # A0
    bb = a1 * a4 - a7 * a8            # A1
    cc = a2 * a3 + a1 * a5 - a6 * a9  # A2
    dd = a2 * a4 - a7 * a9            # A3
    ee = a2 * a5 - a6 * a10           # A4
    ff = -a7 * a10                    # A5

    computeBeta = optSgs in [1, 2]
    if computeBeta:
        beta2_1D = ComputeBeta2(ff, ee, dd, cc, bb, aa)
    else:
        beta2_1D = jnp.ones(nz)
    beta2_3D = jnp.broadcast_to(beta2_1D.reshape(1, 1, nz), (nx, ny, nz))

    # ----------------------------------------------------------
    # WL scalar T_up and T_dn for CwlPrRatio
    # T_up = L^(4/3) * (K'_i * ∂_i c̄  -  α^(4/3)*β2 * K'_i * ∂_i ĉ)
    # T_dn = L^(8/3) * (|∂_i c̄|² - 2*α^(4/3)*β2*(∂_i c̄·∂_i ĉ) + α^(8/3)*β2²*|∂_i ĉ|²)
    # ----------------------------------------------------------
    b6_terms  = (LTH11 * dTHdx_hatd +
                 LTH12 * dTHdy_hatd +
                 LTH13 * dTHdz_hatd)

    c36_terms = (dTHdx_hat * dTHdx_hatd +
                 dTHdy_hat * dTHdy_hatd +
                 dTHdz_hat * dTHdz_hatd)

    T_up = L ** (4 / 3) * (a6_terms - TFR ** (4 / 3) * beta2_3D * b6_terms)

    T_dn = L ** (8 / 3) * (a3_terms
                            - 2 * TFR ** (4 / 3) * beta2_3D * c36_terms
                            + TFR ** (8 / 3) * beta2_3D ** 2 * a8_terms)

    CwlPrRatio_3D = jax.vmap(CwlPrRatio_at_level_k,
                              in_axes=(2, 2), out_axes=2)(T_up, T_dn)

    CwlPrRatio_1D = PlanarMean(CwlPrRatio_3D)

    return CwlPrRatio_3D, CwlPrRatio_1D, beta2_1D
