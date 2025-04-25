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
File: SGS_ScalarLASDD.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: locally-averaged scale-dependent dynamic (LASDD) model
              for scalar transport
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
# Find maximum real root 0 and 5, with focus on 0.5-1.5 range
# ============================================================

@jax.jit
def ComputeBeta2(ff, ee, dd, cc, bb, aa):
    """
    This function solves the polynomial: ff*x^5 + ee*x^4 + dd*x^3 + cc*x^2 + bb*x + aa = 0
    for each vertical level to find the optimal filter ratio (beta2) used in the LASDD model
    for scalar transport.

    Parameters:
    -----------
    ff, ee, dd, cc, bb, aa : jax.numpy.ndarray
        1D arrays containing the polynomial coefficients at each vertical level

    Returns:
    --------
    jax.numpy.ndarray
        1D array of the maximum valid real root for each vertical level,
        constrained between 0 and 5, with default value of 1.0 if no valid root is found
    """

    def find_roots_for_level(k):
        # Construct polynomial coefficients for this level
        coeffs = jnp.array([ff[k], ee[k], dd[k], cc[k], bb[k], aa[k]])

        # Use initial guesses concentrated in the expected range (0.5-1.5)
        # with a few wider points to catch outliers
        guesses = jnp.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5])
        roots = jax.vmap(lambda guess: Roots(coeffs, init_guess=guess))(guesses)

        # Filter valid real roots
        valid_roots = jnp.where(
            (jnp.abs(jnp.imag(roots)) < 1e-10) &
            (jnp.real(roots) > 0) &
            (jnp.real(roots) < 5.0),
            jnp.real(roots),
            jnp.nan
        )

        # Get maximum valid root or default to 1.0
        max_root = jnp.nanmax(valid_roots)
        return jnp.where(jnp.isnan(max_root), 1.0, max_root)

    # Apply to all levels
    return jax.vmap(find_roots_for_level)(jnp.arange(ff.shape[0]))


# ============================================================
# Compute Cs2PrRatio coefficient at vertical level k
# ============================================================

@jax.jit
def Cs2PrRatio_at_level_k(T_up_k, T_dn_k):
    """
    Parameters:
    -----------
    T_up_k : jax.numpy.ndarray
        2D horizontal slice of T_up at level k
    T_dn_k : jax.numpy.ndarray
        2D horizontal slice of T_dn at level k

    Returns:
    --------
    jax.numpy.ndarray
        2D array of the turbulent Prandtl number coefficient at level k
    """

    T_up_F = Imfilter(T_up_k)
    T_dn_F = Imfilter(T_dn_k)

    # Compute Cs2PrRatio
    Cs2PrRatio = T_up_F / T_dn_F

    # Find indices where T_dn_F is too small, Cs2PrRatio < 0, or Cs2PrRatio > 1
    mask = (jnp.abs(T_dn_F) < 1e-10) | (Cs2PrRatio < 0) | (Cs2PrRatio > 1)

    # Apply the mask to set invalid values to zero
    Cs2PrRatio = jnp.where(mask, 0.0, Cs2PrRatio)

    return Cs2PrRatio


# ============================================================
# Main LASDD code for SGS scalar transport model
# ============================================================

@jax.jit
def ScalarLASDD(
        u_, v_, w_,
        u_hat, v_hat, w_hat,
        u_hatd, v_hatd, w_hatd,
        TH,
        dTHdx, dTHdy, dTHdz,
        S, S_hat, S_hatd,
        ZeRo3D):
    """

    Parameters:
    -----------
    u_, v_, w_ : jax.numpy.ndarray
        3D arrays of filtered velocity components
    u_hat, v_hat, w_hat : jax.numpy.ndarray
        3D arrays of level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : jax.numpy.ndarray
        3D arrays of level-2 filtered velocity components
    TH : jax.numpy.ndarray
        3D array of potential temperature
    dTHdx, dTHdy, dTHdz : jax.numpy.ndarray
        3D arrays of potential temperature gradients
    S, S_hat, S_hatd : jax.numpy.ndarray
        3D arrays of strain rate magnitude and its filtered versions
    ZeRo3D : jax.numpy.ndarray
        3D array initialized with zeros

    Returns:
    --------
    Cs2PrRatio_3D : jax.numpy.ndarray
        3D field of Cs2PrRatio
    Cs2PrRatio_1D : jax.numpy.ndarray
        1D profile of Cs2PrRatio
    beta2_1D : jax.numpy.ndarray
        1D profile of beta2 (filter width ratio)
    """

    TH_ = TH.copy()

    # Convert dTHdz to THz (at proper grid locations)
    THz = ZeRo3D.copy()
    THz = THz.at[:, :, 1:nz - 1].set(StagGridAvg(dTHdz[:, :, 1:nz]))
    THz = THz.at[:, :, 0].set(dTHdz[:, :, 0])
    THz = THz.at[:, :, nz - 1].set(dTHdz[:, :, nz - 1])

    # Compute scalar products
    uTH = u_ * TH_
    vTH = v_ * TH_
    wTH = w_ * TH_

    # Apply filtering
    TH_hat = Filtering_Level1(FFT(TH))
    uTH_hat = Filtering_Level1(FFT(uTH))
    vTH_hat = Filtering_Level1(FFT(vTH))
    wTH_hat = Filtering_Level1(FFT(wTH))

    TH_hatd = Filtering_Level2(FFT(TH))
    uTH_hatd = Filtering_Level2(FFT(uTH))
    vTH_hatd = Filtering_Level2(FFT(vTH))
    wTH_hatd = Filtering_Level2(FFT(wTH))

    dTHdx_hat = Filtering_Level1(FFT(dTHdx))
    dTHdy_hat = Filtering_Level1(FFT(dTHdy))
    dTHdz_hat = Filtering_Level1(FFT(THz))

    dTHdx_hatd = Filtering_Level2(FFT(dTHdx))
    dTHdy_hatd = Filtering_Level2(FFT(dTHdy))
    dTHdz_hatd = Filtering_Level2(FFT(THz))

    # Compute and filter strain-gradient products
    SdTHdx = S * dTHdx
    SdTHdy = S * dTHdy
    SdTHdz = S * THz

    SdTHdx_hat = Filtering_Level1(FFT(SdTHdx))
    SdTHdy_hat = Filtering_Level1(FFT(SdTHdy))
    SdTHdz_hat = Filtering_Level1(FFT(SdTHdz))

    SdTHdx_hatd = Filtering_Level2(FFT(SdTHdx))
    SdTHdy_hatd = Filtering_Level2(FFT(SdTHdy))
    SdTHdz_hatd = Filtering_Level2(FFT(SdTHdz))

    # Compute L and Q terms
    LTH11 = uTH_hat - u_hat * TH_hat
    LTH12 = vTH_hat - v_hat * TH_hat
    LTH13 = wTH_hat - w_hat * TH_hat

    QTH11 = uTH_hatd - u_hatd * TH_hatd
    QTH12 = vTH_hatd - v_hatd * TH_hatd
    QTH13 = wTH_hatd - w_hatd * TH_hatd

    # Compute polynomial coefficients
    a2_terms = (LTH11 * SdTHdx_hat +
                LTH12 * SdTHdy_hat +
                LTH13 * SdTHdz_hat)
    a2 = PlanarMean((L ** 2) * a2_terms)

    b2_terms = (LTH11 * S_hat * dTHdx_hat +
                LTH12 * S_hat * dTHdy_hat +
                LTH13 * S_hat * dTHdz_hat)
    b2 = PlanarMean((-L ** 2 * TFR ** 2) * b2_terms)

    c2_terms = (SdTHdx_hat ** 2 +
                SdTHdy_hat ** 2 +
                SdTHdz_hat ** 2)
    c2 = PlanarMean((L ** 4) * c2_terms)

    d2_terms = (SdTHdx_hat * S_hat * dTHdx_hat +
                SdTHdy_hat * S_hat * dTHdy_hat +
                SdTHdz_hat * S_hat * dTHdz_hat)
    d2 = PlanarMean((-2 * L ** 4 * TFR ** 2) * d2_terms)

    e2_terms = ((S_hat * dTHdx_hat) ** 2 +
                (S_hat * dTHdy_hat) ** 2 +
                (S_hat * dTHdz_hat) ** 2)
    e2 = PlanarMean((L ** 4 * TFR ** 4) * e2_terms)

    a4_terms = (QTH11 * SdTHdx_hatd +
                QTH12 * SdTHdy_hatd +
                QTH13 * SdTHdz_hatd)
    a4 = PlanarMean((L ** 2) * a4_terms)

    b4_terms = (QTH11 * S_hatd * dTHdx_hatd +
                QTH12 * S_hatd * dTHdy_hatd +
                QTH13 * S_hatd * dTHdz_hatd)
    b4 = PlanarMean((-L ** 2 * TFR ** 4) * b4_terms)

    c4_terms = (SdTHdx_hatd ** 2 +
                SdTHdy_hatd ** 2 +
                SdTHdz_hatd ** 2)
    c4 = PlanarMean((L ** 4) * c4_terms)

    d4_terms = (SdTHdx_hatd * S_hatd * dTHdx_hatd +
                SdTHdy_hatd * S_hatd * dTHdy_hatd +
                SdTHdz_hatd * S_hatd * dTHdz_hatd)
    d4 = PlanarMean((-2 * L ** 4 * TFR ** 4) * d4_terms)

    e4_terms = ((S_hatd * dTHdx_hatd) ** 2 +
                (S_hatd * dTHdy_hatd) ** 2 +
                (S_hatd * dTHdz_hatd) ** 2)
    e4 = PlanarMean((L ** 4 * TFR ** 8) * e4_terms)

    # Compute polynomial coefficients for beta2
    aa = a2 * c4 - a4 * c2
    bb = -a4 * d2 + b2 * c4
    cc = -c2 * b4 + a2 * d4 - a4 * e2
    dd = b2 * d4 - b4 * d2
    ee = a2 * e4 - b4 * e2
    ff = b2 * e4

    # Compute beta2 for each vertical level
    beta2_1D = ComputeBeta2(ff, ee, dd, cc, bb, aa)

    # Extend beta2 to 3D field
    beta2_3D = jnp.broadcast_to(beta2_1D.reshape(1, 1, nz), (nx, ny, nz))

    # Compute numerator and denominator for Cs2PrRatio
    T_up = ((L ** 2) * a2_terms +
            (-L ** 2 * TFR ** 2) * b2_terms * beta2_3D)
    T_dn = ((L ** 4) * c2_terms +
            (-2 * L ** 4 * TFR ** 2) * d2_terms * beta2_3D +
            (L ** 4 * TFR ** 4) * e2_terms * beta2_3D ** 2)

    # Compute Cs2PrRatio_3D field for all levels using vmap
    Cs2PrRatio_3D = jax.vmap(Cs2PrRatio_at_level_k, in_axes=(2, 2), out_axes=2)(T_up, T_dn)

    # Compute 1D average from the 3D field
    Cs2PrRatio_1D = PlanarMean(Cs2PrRatio_3D)

    return Cs2PrRatio_3D, Cs2PrRatio_1D, beta2_1D
