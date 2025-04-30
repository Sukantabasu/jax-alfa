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
File: Statistics.py
===================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-15
:Description: this file is used to compute various statistics.
"""


# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp
from ..utilities.Utilities import StagGridAvg


# ============================================================
#  Compute planar-averaged statistics
# ============================================================

@jax.jit
def ComputeStats(
        u, v, w,
        TH,
        txy, txz, tyz, qz,
        StatsDict, ResetFlag,
        ZeRo3D):
    """
    Computes spatial averaged-statistics for LES flow variables.

    Parameters:
    -----------
    u, v, w : jnp.ndarray
        Velocity components
    M: jnp.ndarray
        Horizontal wind speed
    TH : jnp.ndarray
        Potential temperature
    txy, txz, tyz : jnp.ndarray
        SGS stress components
    StatsDict : dict
        Dictionary containing accumulated statistics
    ResetFlag : int
        Flag to reset statistics if ResetFlag=1

    Returns:
    --------
    UpdatedStats : dict
        Updated statistics dictionary
    """

    # Extract existing statistics
    u_avg = StatsDict["u"]
    v_avg = StatsDict["v"]
    w_avg = StatsDict["w"]
    M_avg = StatsDict["M"]
    TH_avg = StatsDict["TH"]
    u2_avg = StatsDict["u2"]
    v2_avg = StatsDict["v2"]
    w2_avg = StatsDict["w2"]
    TH2_avg = StatsDict["TH2"]
    uv_avg = StatsDict["uv"]
    uw_avg = StatsDict["uw"]
    vw_avg = StatsDict["vw"]
    uTH_avg = StatsDict["uTH"]
    vTH_avg = StatsDict["vTH"]
    wTH_avg = StatsDict["wTH"]
    txy_avg = StatsDict["txy"]
    txz_avg = StatsDict["txz"]
    tyz_avg = StatsDict["tyz"]
    qz_avg = StatsDict["qz"]
    ustar2_avg = StatsDict["ustar2"]

    # Constants
    Ugal = StatsDict["Ugal"]
    ZeRo1D = StatsDict["ZeRo1D"]

    # Reset statistics
    def ResetStats(_):
        return {
            "u": ZeRo1D,
            "v": ZeRo1D,
            "w": ZeRo1D,
            "M": ZeRo1D,
            "TH": ZeRo1D,
            "u2": ZeRo1D,
            "v2": ZeRo1D,
            "w2": ZeRo1D,
            "TH2": ZeRo1D,
            "uv": ZeRo1D,
            "uw": ZeRo1D,
            "vw": ZeRo1D,
            "uTH": ZeRo1D,
            "vTH": ZeRo1D,
            "wTH": ZeRo1D,
            "txy": ZeRo1D,
            "txz": ZeRo1D,
            "tyz": ZeRo1D,
            "qz": ZeRo1D,
            "ustar2": 0.0,
            "Ugal": Ugal,
            "ZeRo1D": ZeRo1D
        }

    def UpdateStats(_):

        # ------------------------------------------------------------
        # Profiles of mean variables
        # ------------------------------------------------------------

        mu = jnp.mean(u, axis=(0, 1)) + Ugal
        mv = jnp.mean(v, axis=(0, 1))
        mw = jnp.mean(w, axis=(0, 1))

        M = jnp.sqrt((u + Ugal) ** 2 + v ** 2)
        mM = jnp.mean(M, axis=(0, 1))

        mTH = jnp.mean(TH, axis=(0, 1))

        # Updated mean profiles
        new_u_avg = u_avg + mu
        new_v_avg = v_avg + mv
        new_w_avg = w_avg + mw
        new_M_avg = M_avg + mM
        new_TH_avg = TH_avg + mTH

        # ------------------------------------------------------------
        # Profiles of resolved variances and horizontal fluxes
        # ------------------------------------------------------------

        # Compute statistics at each vertical level
        def ComputeLevel1(k):

            # Compute fluctuations
            u_f = u[:, :, k] + Ugal - mu[k]
            v_f = v[:, :, k] - mv[k]
            w_f = w[:, :, k] - mw[k]
            TH_f = TH[:, :, k] - mTH[k]

            # Compute variances and horizontal fluxes
            u2 = jnp.mean(u_f ** 2)
            v2 = jnp.mean(v_f ** 2)
            w2 = jnp.mean(w_f ** 2)
            TH2 = jnp.mean(TH_f ** 2)
            uv = jnp.mean(u_f * v_f)
            uTH = jnp.mean(u_f * TH_f)
            vTH = jnp.mean(v_f * TH_f)

            return u2, v2, w2, TH2, uv, uTH, vTH

        # Apply the computation to all vertical levels
        nz = u.shape[2]
        all_levels = jnp.arange(nz)
        (u2_profile, v2_profile, w2_profile, TH2_profile,
         uv_profile, uTH_profile, vTH_profile) = (
            jax.vmap(ComputeLevel1)(all_levels))

        # Updated variance & horizontal flux profiles
        new_u2_avg = u2_avg + u2_profile
        new_v2_avg = v2_avg + v2_profile
        new_w2_avg = w2_avg + w2_profile
        new_TH2_avg = TH2_avg + TH2_profile
        new_uv_avg = uv_avg + uv_profile
        new_uTH_avg = uTH_avg + uTH_profile
        new_vTH_avg = vTH_avg + vTH_profile

        # ------------------------------------------------------------
        # Resolved flux profiles
        # ------------------------------------------------------------

        # Create staggered velocities using StagGridAvg
        u_stag = ZeRo3D.copy()
        v_stag = ZeRo3D.copy()
        w_stag = w.copy()
        TH_stag = ZeRo3D.copy()

        # Calculate staggered grid variables
        u_stag = u_stag.at[:, :, 1:nz].set(StagGridAvg(u))
        v_stag = v_stag.at[:, :, 1:nz].set(StagGridAvg(v))
        TH_stag = TH_stag.at[:, :, 1:nz].set(StagGridAvg(TH))

        # Set bottom boundary values
        u_stag = u_stag.at[:, :, 0].set(u[:, :, 0] + Ugal)
        v_stag = v_stag.at[:, :, 0].set(v[:, :, 0])
        TH_stag = TH_stag.at[:, :, 0].set(TH[:, :, 0])

        mu_stag = jnp.mean(u_stag, axis=(0, 1)) + Ugal
        mv_stag = jnp.mean(v_stag, axis=(0, 1))
        mw_stag = jnp.mean(w_stag, axis=(0, 1))
        mTH_stag = jnp.mean(TH_stag, axis=(0, 1))

        # Compute fluxes at each vertical level
        def ComputeLevel2(k):
            # Compute fluctuations
            u_stag_f = u_stag[:, :, k] + Ugal - mu_stag[k]
            v_stag_f = v_stag[:, :, k] - mv_stag[k]
            w_stag_f = w_stag[:, :, k] - mw_stag[k]
            TH_stag_f = TH_stag[:, :, k] - mTH_stag[k]

            # Compute fluxes
            uw = jnp.mean(u_stag_f * w_stag_f)
            vw = jnp.mean(v_stag_f * w_stag_f)
            wTH = jnp.mean(w_stag_f * TH_stag_f)

            return uw, vw, wTH

        # Apply the computation to all vertical levels
        (uw_profile, vw_profile, wTH_profile) = jax.vmap(ComputeLevel2)(all_levels)

        # Updated vertical flux profiles
        new_uw_avg = uw_avg + uw_profile
        new_vw_avg = vw_avg + vw_profile
        new_wTH_avg = wTH_avg + wTH_profile

        # ------------------------------------------------------------
        # Profiles of SGS stresses and fluxes
        # ------------------------------------------------------------

        mtxy = jnp.mean(txy, axis=(0, 1))
        mtxz = jnp.mean(txz, axis=(0, 1))
        mtyz = jnp.mean(tyz, axis=(0, 1))
        mqz = jnp.mean(qz, axis=(0, 1))

        # Note: average of momentum fluxes
        ustar2 = jnp.mean(jnp.sqrt(txz[:, :, 0] ** 2 + tyz[:, :, 0] ** 2))

        # Updated mean profiles
        new_txy_avg = txy_avg + mtxy
        new_txz_avg = txz_avg + mtxz
        new_tyz_avg = tyz_avg + mtyz
        new_qz_avg = qz_avg + mqz
        new_ustar2_avg = ustar2_avg + ustar2

        # Create updated statistics dictionary
        return {
            "u": new_u_avg,
            "v": new_v_avg,
            "w": new_w_avg,
            "M": new_M_avg,
            "TH": new_TH_avg,
            "u2": new_u2_avg,
            "v2": new_v2_avg,
            "w2": new_w2_avg,
            "TH2": new_TH2_avg,
            "uv": new_uv_avg,
            "uw": new_uw_avg,
            "vw": new_vw_avg,
            "uTH": new_uTH_avg,
            "vTH": new_vTH_avg,
            "wTH": new_wTH_avg,
            "txy": new_txy_avg,
            "txz": new_txz_avg,
            "tyz": new_tyz_avg,
            "qz": new_qz_avg,
            "ustar2": new_ustar2_avg,
            "Ugal": Ugal,
            "ZeRo1D": ZeRo1D
        }

    # Use JAX's conditional for ResetFlag
    UpdatedStats = jax.lax.cond(
        ResetFlag == 1,
        ResetStats,
        UpdateStats,
        None
    )

    return UpdatedStats


def InitializeStats(nz, Ugal, ZeRo1D):
    """
    Initialize the statistics dictionary with zeros

    Parameters:
    -----------
    nz : int
        Number of vertical grid points
    Ugal : float
        Galilean transformation velocity

    Returns:
    --------
    stats_dict : dict
        Initialized statistics dictionary
    """

    StatsDict = {
        "u": ZeRo1D,
        "v": ZeRo1D,
        "w": ZeRo1D,
        "M": ZeRo1D,
        "TH": ZeRo1D,
        "u2": ZeRo1D,
        "v2": ZeRo1D,
        "w2": ZeRo1D,
        "TH2": ZeRo1D,
        "uv": ZeRo1D,
        "uw": ZeRo1D,
        "vw": ZeRo1D,
        "uTH": ZeRo1D,
        "vTH": ZeRo1D,
        "wTH": ZeRo1D,
        "txy": ZeRo1D,
        "txz": ZeRo1D,
        "tyz": ZeRo1D,
        "qz": ZeRo1D,
        "ustar2": 0.0,
        "Ugal": Ugal,
        "ZeRo1D": ZeRo1D
    }

    return StatsDict
