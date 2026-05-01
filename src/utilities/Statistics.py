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
:Date: 2025-5-3
:Description: this file is used to compute various statistics.
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp
from ..utilities.Utilities import StagGridAvg

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *


# ============================================================
#  Compute planar-averaged statistics
# ============================================================

@jax.jit
def ComputeStats(
        u, v, w, TH,
        dudz, dvdz, dTHdz,
        M_sfc_loc, ustar,
        txy, txz, tyz, qz,
        Cs2_1D_avg1, Cs2_1D_avg2,
        Cs2PrRatio_1D,
        beta1_1D, beta2_1D,
        StatsDict, ResetFlag,
        ZeRo3D):
    """
    Computes spatial averaged-statistics for LES flow variables.

    Parameters:
    -----------
    u, v, w : ndarray
        Velocity components
    TH : ndarray
        Potential temperature
    dudz, dvdz : ndarray
        Velocity gradients
    dTHdz : ndarray
        Potential temperature gradients
    M_sfc_loc : ndarray of shape (nx, ny)
        Wind speed at z = 0.5 * dz
    ustar : ndarray of shape (nx, ny)
        Friction velocity
    txy, txz, tyz : ndarray
        SGS stress components
    qz : ndarray
        SGS heat flux in z-direction
    Cs2_1D_avg1 : ndarray of shape (nz)
        Smagorinsky coefficient
    Cs2_1D_avg2 : ndarray of shape (nz)
        Smagorinsky coefficient
    Cs2PrRatio_1D : ndarray of shape (nz)
        Ratio of Smagorinsky coefficient and SGS Prandtl number
    beta1_1D : ndarray of shape (nz)
        Beta coefficient for momentum
    beta2_1D : ndarray of shape (nz)
        Beta coefficient for scalar
    StatsDict : dict
        Dictionary containing accumulated statistics
    ResetFlag : int
        Flag to reset statistics if ResetFlag=1
    ZeRo3D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    UpdatedStats : dict
        Updated statistics dictionary
    """

    # Extract existing statistics

    # Mean profiles
    U_avg = StatsDict["U"]; V_avg = StatsDict["V"]
    W_avg = StatsDict["W"]; TH_avg = StatsDict["TH"]

    # Mean gradients
    dUdz_avg = StatsDict["dUdz"]; dVdz_avg = StatsDict["dVdz"]
    dTHdz_avg = StatsDict["dTHdz"]

    # Resolved variances
    u2_avg = StatsDict["u2"]; v2_avg = StatsDict["v2"]
    w2_avg = StatsDict["w2"]; TH2_avg = StatsDict["TH2"]

    # Resolved fluxes
    uv_avg = StatsDict["uv"]; uw_avg = StatsDict["uw"]
    vw_avg = StatsDict["vw"]; uTH_avg = StatsDict["uTH"]
    vTH_avg = StatsDict["vTH"]; wTH_avg = StatsDict["wTH"]

    # Surface terms
    M_sfc_avg = StatsDict["M_sfc"]; ustar_avg = StatsDict["ustar"]

    # SGS terms
    txy_avg = StatsDict["txy"]; txz_avg = StatsDict["txz"]
    tyz_avg = StatsDict["tyz"]; qz_avg = StatsDict["qz"]

    # SGS coefficients
    Cs2_1_avg = StatsDict["Cs2_1"]; Cs2_2_avg = StatsDict["Cs2_2"]
    Cs2PrRatio_avg = StatsDict["Cs2PrRatio"]
    Beta1_avg = StatsDict["Beta1"]; Beta2_avg = StatsDict["Beta2"]

    # Constants
    Ugal = StatsDict["Ugal"]; ZeRo1D = StatsDict["ZeRo1D"]

    # Reset statistics function
    def ResetStats(_):
        return {
            "U": ZeRo1D, "V": ZeRo1D, "W": ZeRo1D, "TH": ZeRo1D,
            "dUdz": ZeRo1D, "dVdz": ZeRo1D, "dTHdz": ZeRo1D,
            "u2": ZeRo1D, "v2": ZeRo1D, "w2": ZeRo1D, "TH2": ZeRo1D,
            "uv": ZeRo1D, "uw": ZeRo1D, "vw": ZeRo1D,
            "uTH": ZeRo1D, "vTH": ZeRo1D, "wTH": ZeRo1D,
            "txy": ZeRo1D, "txz": ZeRo1D, "tyz": ZeRo1D,
            "qz": ZeRo1D,
            "M_sfc": 0.0, "ustar": 0.0,
            "Cs2_1": ZeRo1D, "Cs2_2": ZeRo1D,
            "Cs2PrRatio": ZeRo1D,
            "Beta1": ZeRo1D, "Beta2": ZeRo1D,
            "Ugal": Ugal, "ZeRo1D": ZeRo1D
        }

    # Update statistics function
    def UpdateStats(_):
        # ------------------------------------------------------------
        # Profiles of mean variables
        # ------------------------------------------------------------
        mU = (jnp.mean(u, axis=(0, 1)) + Ugal) * u_scale
        mV = jnp.mean(v, axis=(0, 1)) * u_scale
        mW = jnp.mean(w, axis=(0, 1)) * u_scale
        mTH = jnp.mean(TH, axis=(0, 1)) * TH_scale

        mdUdz = jnp.mean(dudz, axis=(0, 1)) * (u_scale / z_scale)
        mdVdz = jnp.mean(dvdz, axis=(0, 1)) * (u_scale / z_scale)
        mdTHdz = jnp.mean(dTHdz, axis=(0, 1)) * (TH_scale / z_scale)

        # Updated mean profiles
        new_U_avg = U_avg + mU
        new_V_avg = V_avg + mV
        new_W_avg = W_avg + mW
        new_TH_avg = TH_avg + mTH
        new_dUdz_avg = dUdz_avg + mdUdz
        new_dVdz_avg = dVdz_avg + mdVdz
        new_dTHdz_avg = dTHdz_avg + mdTHdz

        # ------------------------------------------------------------
        # Profiles of resolved variances and horizontal fluxes
        # ------------------------------------------------------------
        def ComputeLevel1(k):
            """Compute variances and horizontal fluxes at vertical level k."""
            # Compute fluctuations
            u_f = u[:, :, k] + Ugal - mU[k]
            v_f = v[:, :, k] - mV[k]
            w_f = w[:, :, k] - mW[k]
            TH_f = TH[:, :, k] - mTH[k]

            # Compute variances and horizontal fluxes
            u2 = jnp.mean(u_f ** 2) * (u_scale ** 2)
            v2 = jnp.mean(v_f ** 2) * (u_scale ** 2)
            w2 = jnp.mean(w_f ** 2) * (u_scale ** 2)
            TH2 = jnp.mean(TH_f ** 2) * (TH_scale ** 2)
            uv = jnp.mean(u_f * v_f) * (u_scale ** 2)
            uTH = jnp.mean(u_f * TH_f) * (u_scale * TH_scale)
            vTH = jnp.mean(v_f * TH_f) * (u_scale * TH_scale)

            return u2, v2, w2, TH2, uv, uTH, vTH

        # Apply the computation to all vertical levels
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

        # Compute staggered means
        mu_stag = (jnp.mean(u_stag, axis=(0, 1)) + Ugal) * u_scale
        mv_stag = jnp.mean(v_stag, axis=(0, 1)) * u_scale
        mw_stag = jnp.mean(w_stag, axis=(0, 1)) * u_scale
        mTH_stag = jnp.mean(TH_stag, axis=(0, 1)) * TH_scale

        def ComputeLevel2(k):
            """Compute vertical fluxes at vertical level k."""
            # Compute fluctuations
            u_stag_f = u_stag[:, :, k] + Ugal - mu_stag[k]
            v_stag_f = v_stag[:, :, k] - mv_stag[k]
            w_stag_f = w_stag[:, :, k] - mw_stag[k]
            TH_stag_f = TH_stag[:, :, k] - mTH_stag[k]

            # Compute fluxes
            uw = jnp.mean(u_stag_f * w_stag_f) * (u_scale ** 2)
            vw = jnp.mean(v_stag_f * w_stag_f) * (u_scale ** 2)
            wTH = jnp.mean(w_stag_f * TH_stag_f) * (u_scale * TH_scale)

            return uw, vw, wTH

        # Apply the computation to all vertical levels
        (uw_profile, vw_profile, wTH_profile) = jax.vmap(ComputeLevel2)(
            all_levels)

        # Updated vertical flux profiles
        new_uw_avg = uw_avg + uw_profile
        new_vw_avg = vw_avg + vw_profile
        new_wTH_avg = wTH_avg + wTH_profile

        # ------------------------------------------------------------
        # Profiles of SGS stresses and fluxes
        # ------------------------------------------------------------
        mtxy = jnp.mean(txy, axis=(0, 1)) * (u_scale ** 2)
        mtxz = jnp.mean(txz, axis=(0, 1)) * (u_scale ** 2)
        mtyz = jnp.mean(tyz, axis=(0, 1)) * (u_scale ** 2)
        mqz = jnp.mean(qz, axis=(0, 1)) * (u_scale * TH_scale)

        # ------------------------------------------------------------
        # Updated SGS profiles
        # ------------------------------------------------------------
        new_txy_avg = txy_avg + mtxy
        new_txz_avg = txz_avg + mtxz
        new_tyz_avg = tyz_avg + mtyz
        new_qz_avg = qz_avg + mqz

        # ------------------------------------------------------------
        # Surface variables
        # ------------------------------------------------------------
        mM_sfc = jnp.mean(M_sfc_loc)
        # Note: average of momentum fluxes
        mustar = jnp.sqrt(jnp.mean(ustar ** 2)) * u_scale

        # ------------------------------------------------------------
        # Updated surface variables
        # ------------------------------------------------------------
        new_M_sfc_avg = M_sfc_avg + mM_sfc
        new_ustar_avg = ustar_avg + mustar

        # ------------------------------------------------------------
        # Updated SGS coefficients
        # ------------------------------------------------------------
        new_Cs2_1_avg = Cs2_1_avg + Cs2_1D_avg1
        new_Cs2_2_avg = Cs2_2_avg + Cs2_1D_avg2
        new_Cs2PrRatio_avg = Cs2PrRatio_avg + Cs2PrRatio_1D
        new_Beta1_avg = Beta1_avg + beta1_1D
        new_Beta2_avg = Beta2_avg + beta2_1D

        # Create updated statistics dictionary
        return {
            "U": new_U_avg, "V": new_V_avg, "W": new_W_avg, "TH": new_TH_avg,
            "dUdz": new_dUdz_avg, "dVdz": new_dVdz_avg,
            "dTHdz": new_dTHdz_avg,
            "u2": new_u2_avg, "v2": new_v2_avg, "w2": new_w2_avg,
            "TH2": new_TH2_avg,
            "uv": new_uv_avg, "uw": new_uw_avg, "vw": new_vw_avg,
            "uTH": new_uTH_avg, "vTH": new_vTH_avg, "wTH": new_wTH_avg,
            "txy": new_txy_avg, "txz": new_txz_avg, "tyz": new_tyz_avg,
            "qz": new_qz_avg,
            "M_sfc": new_M_sfc_avg, "ustar": new_ustar_avg,
            "Cs2_1": new_Cs2_1_avg, "Cs2_2": new_Cs2_2_avg,
            "Cs2PrRatio": new_Cs2PrRatio_avg,
            "Beta1": new_Beta1_avg, "Beta2": new_Beta2_avg,
            "Ugal": Ugal, "ZeRo1D": ZeRo1D
        }

    # Use JAX's conditional for ResetFlag
    UpdatedStats = jax.lax.cond(
        ResetFlag == 1,
        ResetStats,
        UpdateStats,
        None
    )

    return UpdatedStats


def InitializeStats(ZeRo1D):
    """
    Initialize the statistics dictionary with zeros.

    Parameters:
    -----------
    ZeRo1D : ndarray
        Pre-allocated zero array

    Returns:
    --------
    StatsDict : dict
        Initialized statistics dictionary
    """
    StatsDict = {
        # Mean profiles
        "U": ZeRo1D, "V": ZeRo1D, "W": ZeRo1D, "TH": ZeRo1D,

        # Mean gradients
        "dUdz": ZeRo1D, "dVdz": ZeRo1D, "dTHdz": ZeRo1D,

        # Resolved variances
        "u2": ZeRo1D, "v2": ZeRo1D, "w2": ZeRo1D, "TH2": ZeRo1D,

        # Resolved fluxes
        "uv": ZeRo1D, "uw": ZeRo1D, "vw": ZeRo1D,
        "uTH": ZeRo1D, "vTH": ZeRo1D, "wTH": ZeRo1D,

        # SGS terms
        "txy": ZeRo1D, "txz": ZeRo1D, "tyz": ZeRo1D,
        "qz": ZeRo1D,

        # Surface terms
        "M_sfc": 0.0, "ustar": 0.0,

        # SGS coefficients
        "Cs2_1": ZeRo1D, "Cs2_2": ZeRo1D, "Cs2PrRatio": ZeRo1D,
        "Beta1": ZeRo1D, "Beta2": ZeRo1D,

        # Constants
        "Ugal": Ugal, "ZeRo1D": ZeRo1D
    }

    return StatsDict