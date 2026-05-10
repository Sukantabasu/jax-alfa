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
File: DynamicSGS_LASDD.py
=========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: locally-averaged scale-dependent dynamic (LASDD) model
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
def ComputeBeta1(ff, ee, dd, cc, bb, aa):
    """
    This function solves the polynomial:
    ff*x^5 + ee*x^4 + dd*x^3 + cc*x^2 + bb*x + aa = 0
    for each vertical level to find the optimal parameter beta1
    used in the LASDD SGS model

    Parameters:
    -----------
    ff, ee, dd, cc, bb, aa : ndarray
        1D arrays containing the polynomial coefficients at each vertical level

    Returns:
    --------
    beta1 : ndarray
        1D array of the maximum valid real root for each vertical level
    """

    def find_roots_for_level(k):
        # Construct polynomial coefficients for this level
        coeffs = jnp.array([ff[k], ee[k], dd[k], cc[k], bb[k], aa[k]])

        # Use initial guesses concentrated in the expected range (0.5-1.5)
        # with a few wider points to catch outliers
        guesses = jnp.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5])
        roots = jax.vmap(lambda guess:
                         Roots(coeffs, init_guess=guess))(guesses)

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


# ===================================================
# Compute Smagorinsky coefficient at vertical level k
# ===================================================

@jax.jit
def Cs2_at_level_k(LM_k, MM_k):
    """
    Parameters:
    -----------
    LM_k : jax.numpy.ndarray
        2D horizontal slice of LM at level k
    MM_k : jax.numpy.ndarray
        2D horizontal slice of MM at level k

    Returns:
    --------
    Cs2: ndarray
        2D array of the squared Smagorinsky coefficient at level k
    """

    LMx = Imfilter(LM_k)
    MMx = Imfilter(MM_k)

    # Compute Temp with division
    Cs2 = LMx / MMx

    # Find indices where MMx is too small, Cs2 < 0, or Cs2 > 1
    mask = (jnp.abs(MMx) < 1e-10) | (Cs2 < 0) | (Cs2 > 1)

    # Apply the mask to set invalid values to zero
    Cs2 = jnp.where(mask, 0.0, Cs2)

    return Cs2


# ============================================================
# Main LASDD code
# ============================================================

@jax.jit
def LASDD(
        u, v, w,
        S11, S22, S33,
        S12, S13, S23,
        S,
        ZeRo3D):
    """
    Parameters:
    -----------
    u, v, w : ndarray
        Velocity components
    S11, S22, S33 : ndarray
        Normal strain rate components
    S12, S13, S23 : ndarray
        Shear strain rate components
    S : ndarray
        Strain rate magnitude
    ZeRo3D : ndarray
        Pre-allocated zero arrays

    Returns:
    --------
    u_, v_, w_ : ndarray
        Interpolated velocity components
    u_hat, v_hat, w_hat : ndarray
        Level-1 filtered velocity components
    u_hatd, v_hatd, w_hatd : ndarray
        Level-2 filtered velocity components
    S_hat, S_hatd : ndarray
       Filtered strain rate magnitudes
    Cs2_3D : ndarray
        Cs2 field
    Cs2_1D_avg1 : ndarray
        1D profile of Cs2 (method 1: square of mean of sqrt)
    Cs2_1D_avg2 : ndarray
        1D profile of Cs2 (method 2: direct mean)
    beta1_1D : ndarray
        1D profile of beta1
    """

    u_ = u.copy()
    v_ = v.copy()
    w_ = ZeRo3D.copy()
    w_ = w_.at[:, :, 1:nz - 1].set(StagGridAvg(w[:, :, 1:nz]))
    w_ = w_.at[:, :, 0].set(0.5 * w[:, :, 1])
    w_ = w_.at[:, :, nz - 1].set(w[:, :, nz - 2])

    # Compute squared terms
    uu, vv, ww = u_ ** 2, v_ ** 2, w_ ** 2
    uv, uw, vw = u_ * v_, u_ * w_, v_ * w_

    # Apply filtering
    u_hat = Filtering_Level1(FFT(u_))
    v_hat = Filtering_Level1(FFT(v_))
    w_hat = Filtering_Level1(FFT(w_))
    uu_hat = Filtering_Level1(FFT(uu))
    vv_hat = Filtering_Level1(FFT(vv))
    ww_hat = Filtering_Level1(FFT(ww))
    uv_hat = Filtering_Level1(FFT(uv))
    uw_hat = Filtering_Level1(FFT(uw))
    vw_hat = Filtering_Level1(FFT(vw))

    u_hatd = Filtering_Level2(FFT(u_))
    v_hatd = Filtering_Level2(FFT(v_))
    w_hatd = Filtering_Level2(FFT(w_))
    uu_hatd = Filtering_Level2(FFT(uu))
    vv_hatd = Filtering_Level2(FFT(vv))
    ww_hatd = Filtering_Level2(FFT(ww))
    uv_hatd = Filtering_Level2(FFT(uv))
    uw_hatd = Filtering_Level2(FFT(uw))
    vw_hatd = Filtering_Level2(FFT(vw))

    # Filter strain rate components
    S11_hat = Filtering_Level1(FFT(S11))
    S22_hat = Filtering_Level1(FFT(S22))
    S33_hat = Filtering_Level1(FFT(S33))
    S12_hat = Filtering_Level1(FFT(S12))
    S13_hat = Filtering_Level1(FFT(S13))
    S23_hat = Filtering_Level1(FFT(S23))

    S11_hatd = Filtering_Level2(FFT(S11))
    S22_hatd = Filtering_Level2(FFT(S22))
    S33_hatd = Filtering_Level2(FFT(S33))
    S12_hatd = Filtering_Level2(FFT(S12))
    S13_hatd = Filtering_Level2(FFT(S13))
    S23_hatd = Filtering_Level2(FFT(S23))

    # Compute filtered strain rate magnitudes
    S_hat = jnp.sqrt(2 * (S11_hat ** 2 + S22_hat ** 2 + S33_hat ** 2 +
                          2 * S12_hat ** 2 +
                          2 * S13_hat ** 2 +
                          2 * S23_hat ** 2))
    S_hatd = jnp.sqrt(2 * (S11_hatd ** 2 + S22_hatd ** 2 + S33_hatd ** 2 +
                           2 * S12_hatd ** 2 +
                           2 * S13_hatd ** 2 +
                           2 * S23_hatd ** 2))

    # Compute and filter strain rate products
    SS11_hat = Filtering_Level1(FFT(S * S11))
    SS22_hat = Filtering_Level1(FFT(S * S22))
    SS33_hat = Filtering_Level1(FFT(S * S33))
    SS12_hat = Filtering_Level1(FFT(S * S12))
    SS13_hat = Filtering_Level1(FFT(S * S13))
    SS23_hat = Filtering_Level1(FFT(S * S23))

    SS11_hatd = Filtering_Level2(FFT(S * S11))
    SS22_hatd = Filtering_Level2(FFT(S * S22))
    SS33_hatd = Filtering_Level2(FFT(S * S33))
    SS12_hatd = Filtering_Level2(FFT(S * S12))
    SS13_hatd = Filtering_Level2(FFT(S * S13))
    SS23_hatd = Filtering_Level2(FFT(S * S23))

    # Compute L and Q tensors
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

    a1_terms = (L11 * SS11_hat + L22 * SS22_hat + L33 * SS33_hat +
                2 * (L12 * SS12_hat + L13 * SS13_hat + L23 * SS23_hat))
    a2_terms = (Q11 * SS11_hatd + Q22 * SS22_hatd + Q33 * SS33_hatd +
                2 * (Q12 * SS12_hatd + Q13 * SS13_hatd + Q23 * SS23_hatd))

    a1 = PlanarMean(2 * (L ** 2) * a1_terms)
    a2 = PlanarMean(2 * (L ** 2) * a2_terms)

    b1_terms = (L11 * S11_hat + L22 * S22_hat + L33 * S33_hat +
                2 * (L12 * S12_hat + L13 * S13_hat + L23 * S23_hat))
    b2_terms = (Q11 * S11_hatd + Q22 * S22_hatd + Q33 * S33_hatd +
                2 * (Q12 * S12_hatd + Q13 * S13_hatd + Q23 * S23_hatd))

    b1 = PlanarMean(2 * (L ** 2) * (TFR ** 2) * S_hat * b1_terms)
    b2 = PlanarMean(2 * (L ** 2) * (TFR ** 4) * S_hatd * b2_terms)

    c1_terms = (SS11_hat ** 2 + SS22_hat ** 2 + SS33_hat ** 2 +
                2 * (SS12_hat ** 2 + SS13_hat ** 2 + SS23_hat ** 2))
    c2_terms = (SS11_hatd ** 2 + SS22_hatd ** 2 + SS33_hatd ** 2 +
                2 * (SS12_hatd ** 2 + SS13_hatd ** 2 + SS23_hatd ** 2))

    c1 = PlanarMean((2 * L ** 2) ** 2 * c1_terms)
    c2 = PlanarMean((2 * L ** 2) ** 2 * c2_terms)

    d1_terms = (S11_hat ** 2 + S22_hat ** 2 + S33_hat ** 2 +
                2 * (S12_hat ** 2 + S13_hat ** 2 + S23_hat ** 2))
    d2_terms = (S11_hatd ** 2 + S22_hatd ** 2 + S33_hatd ** 2 +
                2 * (S12_hatd ** 2 + S13_hatd ** 2 + S23_hatd ** 2))

    d1 = PlanarMean((4 * L ** 4) * (TFR ** 4) * (S_hat ** 2) * d1_terms)
    d2 = PlanarMean((4 * L ** 4) * (TFR ** 8) * (S_hatd ** 2) * d2_terms)

    e1_terms = (S11_hat * SS11_hat +
                S22_hat * SS22_hat +
                S33_hat * SS33_hat +
                2 * (S12_hat * SS12_hat +
                     S13_hat * SS13_hat +
                     S23_hat * SS23_hat))
    e2_terms = (S11_hatd * SS11_hatd +
                S22_hatd * SS22_hatd +
                S33_hatd * SS33_hatd +
                2 * (S12_hatd * SS12_hatd +
                     S13_hatd * SS13_hatd +
                     S23_hatd * SS23_hatd))

    e1 = PlanarMean((8 * L ** 4) * (TFR ** 2) * S_hat * e1_terms)
    e2 = PlanarMean((8 * L ** 4) * (TFR ** 4) * S_hatd * e2_terms)

    # Compute polynomial coefficients
    aa = a1 * c2 - a2 * c1
    bb = a2 * e1 - b1 * c2
    cc = b2 * c1 - a1 * e2 - a2 * d1
    dd = b1 * e2 - b2 * e1
    ee = a1 * d2 + b2 * d1
    ff = -b1 * d2

    beta1_1D = ComputeBeta1(ff, ee, dd, cc, bb, aa)
    # Extend beta1 to 3D field
    beta1_3D = jnp.broadcast_to(beta1_1D.reshape(1, 1, nz), (nx, ny, nz))

    # Compute M terms
    T1 = 2 * L ** 2
    T2 = 2 * (TFR * L) ** 2
    M11 = T1 * SS11_hat - T2 * beta1_3D * S_hat * S11_hat
    M22 = T1 * SS22_hat - T2 * beta1_3D * S_hat * S22_hat
    M33 = T1 * SS33_hat - T2 * beta1_3D * S_hat * S33_hat
    M12 = T1 * SS12_hat - T2 * beta1_3D * S_hat * S12_hat
    M13 = T1 * SS13_hat - T2 * beta1_3D * S_hat * S13_hat
    M23 = T1 * SS23_hat - T2 * beta1_3D * S_hat * S23_hat

    # Compute LM and MM terms
    LM = ((L11 * M11 +
          L22 * M22 +
          L33 * M33) +
          2 * (L12 * M12 +
               L13 * M13 +
               L23 * M23))

    MM = (M11 ** 2 +
          M22 ** 2 +
          M33 ** 2 +
          2 * (M12 ** 2 +
               M13 ** 2 +
               M23 ** 2))

    # Compute Cs2_3D field for all levels using vmap
    Cs2_3D = jax.vmap(Cs2_at_level_k, in_axes=(2, 2), out_axes=2)(LM, MM)

    # Compute 1D averages from the 3D field
    # First compute sqrt(Cs2_3D) for each level and then square the mean
    Cs2_1D_avg1 = PlanarMean(jnp.sqrt(Cs2_3D)) ** 2
    # Compute simple mean for each level
    Cs2_1D_avg2 = PlanarMean(Cs2_3D)

    return (u_, v_, w_,
            u_hat, v_hat, w_hat,
            u_hatd, v_hatd, w_hatd,
            S_hat, S_hatd,
            Cs2_3D, Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D)
