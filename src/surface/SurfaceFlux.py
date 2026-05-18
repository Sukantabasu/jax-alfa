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
File: SurfaceFlux.py
==============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: surface flux calculation module.
              Supports constant flux (optSurfBC=0), time-varying flux
              (optSurfBC=1), and prescribed surface temperature (optSurfBC=2),
              each in homogeneous (optSurfFlux=0) and heterogeneous
              (optSurfFlux=1) flavours.

              Sign convention: qz = -u* x th*
              Stable BL: qz < 0 (downward), th* > 0
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


# ============================================================
#  Monin-Obukhov similarity functions
# ============================================================

@jax.jit
def MOSTstable(z_over_L):
    """
    Compute Monin-Obukhov stability functions for stable conditions

    Parameters:
    -----------
    z_over_L : jnp.ndarray
        Stability parameter z/L (height / Obukhov length)

    Returns:
    --------
    psi_m : jnp.ndarray
        Stability function for momentum
    psi_h : jnp.ndarray
        Stability function for heat
    fi_m : jnp.ndarray
        Normalized gradient function for momentum
    fi_h : jnp.ndarray
        Normalized gradient function for heat
    """

    psi_m = 5.0 * z_over_L
    psi_h = 5.0 * z_over_L

    fi_m = 1.0 + psi_m
    fi_h = 1.0 + psi_h

    return psi_m, psi_h, fi_m, fi_h


def MOSTunstable(z_over_L):
    """
    Compute Monin-Obukhov stability functions for unstable conditions

    Parameters:
    -----------
    z_over_L : jnp.ndarray of shape (nx, ny)
        Stability parameter z/L (height / Obukhov length)

    Returns:
    --------
    psi2D_m : jnp.ndarray of shape (nx, ny)
        Stability function for momentum
    psi2D_h : jnp.ndarray of shape (nx, ny)
        Stability function for heat
    fi2D_m : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for momentum
    fi2D_h : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for heat
    """

    x = (1 - 15 * z_over_L) ** 0.25

    psi2D_m = (-2 * jnp.log(0.5 * (1 + x)) - jnp.log(0.5 * (1 + x ** 2))
               + 2 * jnp.arctan(x) - jnp.pi / 2)
    psi2D_h = -2 * jnp.log(0.5 * (1 + x ** 2))

    fi2D_m = 1.0 / x
    fi2D_h = (1.0 / x) ** 2
    return psi2D_m, psi2D_h, fi2D_m, fi2D_h


# ============================================================
#  Shared helper: update MOST stability functions
# ============================================================

def _update_MOSTfunctions(ustar, qz_sfc_avg, TH_ref, psi2D_m, psi2D_m0,
                          psi2D_h, psi2D_h0):
    """
    Compute Obukhov length and update all MOST stability functions.
    Used internally by all six surface flux variants.

    Returns updated (psi2D_m, psi2D_m0, psi2D_h, psi2D_h0, fi2D_m, fi2D_h)
    and invOB.
    """
    invOB = -(vonk * g_nondim * qz_sfc_avg) / ((ustar ** 3) * TH_ref)
    is_stable = qz_sfc_avg <= 0

    z1_over_L   = (0.5 * dz) * invOB
    z0m_over_L  = (z0m / z_scale) * invOB
    z0T_over_L  = (z0T / z_scale) * invOB

    psi2D_m, psi2D_h, fi2D_m, fi2D_h = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z1_over_L),
        lambda _: MOSTunstable(z1_over_L),
        operand=None)

    psi2D_m0, _, _, _ = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0m_over_L),
        lambda _: MOSTunstable(z0m_over_L),
        operand=None)

    _, psi2D_h0, _, _ = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0T_over_L),
        lambda _: MOSTunstable(z0T_over_L),
        operand=None)

    MOSTfunctions = (psi2D_m, psi2D_m0, psi2D_h, psi2D_h0, fi2D_m, fi2D_h)
    return MOSTfunctions, invOB


# ============================================================
#  optSurfBC = 0 : constant prescribed heat flux
# ============================================================

@jax.jit
def SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions):
    """
    optSurfFlux=0, optSurfBC=0: homogeneous surface, constant heat flux.

    Parameters:
    -----------
    u, v : jnp.ndarray of shape (nx, ny, nz)
    TH   : jnp.ndarray of shape (nx, ny, nz)
    MOSTfunctions : tuple of six (nx, ny) arrays

    Returns:
    --------
    M_sfc_loc    : (nx, ny) surface wind speed
    ustar        : (nx, ny) friction velocity
    qz_sfc_avg   : scalar, non-dimensional surface heat flux (= qz = -u* x th*)
    invOB        : (nx, ny) inverse Obukhov length
    MOSTfunctions: updated tuple
    """

    One2D = jnp.ones((nx, ny))

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = jnp.mean(qz_sfc)

    M_sfc_avg = jnp.mean(jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2))
    M_sfc_loc = M_sfc_avg * One2D

    # TH is anomaly; add T_0 to get absolute temperature for Obukhov length.
    TH_ref = (jnp.mean(TH[:, :, 0]) + T_0_nondim) * One2D

    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_ref, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions


@jax.jit
def SurfaceFlux_HeterogeneousConstantFlux(u, v, TH, MOSTfunctions):
    """
    optSurfFlux=1, optSurfBC=0: heterogeneous surface, constant heat flux.

    Returns:
    --------
    M_sfc_loc    : (nx, ny)
    ustar        : (nx, ny)
    qz_sfc_avg   : scalar, non-dimensional surface heat flux
    invOB        : (nx, ny)
    MOSTfunctions: updated tuple
    """

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = jnp.mean(qz_sfc)

    M_sfc_loc = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)
    # TH is anomaly; add T_0 for absolute temperature for Obukhov length.
    TH_ref = TH[:, :, 0] + T_0_nondim

    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_ref, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions


# ============================================================
#  optSurfBC = 1 : time-varying prescribed heat flux
# ============================================================

@jax.jit
def SurfaceFlux_HomogeneousVaryingFlux(u, v, TH, qz_sfc_t, MOSTfunctions):
    """
    optSurfFlux=0, optSurfBC=1: homogeneous surface, time-varying heat flux.

    Parameters:
    -----------
    qz_sfc_t : scalar JAX value, non-dimensional heat flux at current timestep
               (loaded from SurfaceBC.npz series, already non-dimensionalised)

    Returns:
    --------
    M_sfc_loc    : (nx, ny)
    ustar        : (nx, ny)
    qz_sfc_2D    : (nx, ny) spatially uniform flux field
    qz_sfc_avg   : scalar
    invOB        : (nx, ny)
    MOSTfunctions: updated tuple
    """

    One2D = jnp.ones((nx, ny))

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = qz_sfc_t
    qz_sfc_2D  = qz_sfc_t * One2D

    M_sfc_avg = jnp.mean(jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2))
    M_sfc_loc = M_sfc_avg * One2D

    # TH is anomaly; add T_0 for absolute temperature for Obukhov length.
    TH_ref = (jnp.mean(TH[:, :, 0]) + T_0_nondim) * One2D

    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_ref, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_2D, qz_sfc_avg, invOB, MOSTfunctions


@jax.jit
def SurfaceFlux_HeterogeneousVaryingFlux(u, v, TH, qz_sfc_t, MOSTfunctions):
    """
    optSurfFlux=1, optSurfBC=1: heterogeneous surface, time-varying heat flux.

    Parameters:
    -----------
    qz_sfc_t : scalar JAX value, non-dimensional heat flux at current timestep

    Returns:
    --------
    M_sfc_loc    : (nx, ny)
    ustar        : (nx, ny)
    qz_sfc_2D    : (nx, ny)
    qz_sfc_avg   : scalar
    invOB        : (nx, ny)
    MOSTfunctions: updated tuple
    """

    One2D = jnp.ones((nx, ny))

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = qz_sfc_t
    qz_sfc_2D  = qz_sfc_t * One2D

    M_sfc_loc = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)
    # TH is anomaly; add T_0 for absolute temperature for Obukhov length.
    TH_ref = TH[:, :, 0] + T_0_nondim

    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_ref, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_2D, qz_sfc_avg, invOB, MOSTfunctions


# ============================================================
#  optSurfBC = 2 : time-varying prescribed surface temperature
# ============================================================

@jax.jit
def SurfaceFlux_HomogeneousPrescribedTemperature(u, v, TH, TH_sfc_t,
                                                  MOSTfunctions):
    """
    optSurfFlux=0, optSurfBC=2: homogeneous surface, prescribed T_s(t).

    The surface heat flux is diagnosed from MOST:
        qz = u* x vonk x (TH_s - TH_air) / denom_h
    consistent with qz = -u* x th*, where th* = vonk x (TH_air - TH_s) / denom_h.

    Parameters:
    -----------
    TH_sfc_t : scalar JAX value, non-dimensional surface temperature anomaly
               (theta_sfc - T_0) / TH_scale at current timestep, as returned
               by Initialize_SurfaceBC for optSurfBC=2

    Returns:
    --------
    M_sfc_loc    : (nx, ny)
    ustar        : (nx, ny)
    qz_sfc_2D    : (nx, ny) diagnosed surface heat flux
    qz_sfc_avg   : scalar
    invOB        : (nx, ny)
    MOSTfunctions: updated tuple
    """

    One2D = jnp.ones((nx, ny))

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    # Planar-mean surface wind speed
    M_sfc_avg = jnp.mean(jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2))
    M_sfc_loc = M_sfc_avg * One2D

    # TH is stored as anomaly (TH - T_0); TH_sfc_t is also an anomaly from
    # Initialize_SurfaceBC. Both are anomalies so the difference is direct.
    TH_air_anom_avg = jnp.mean(TH[:, :, 0])
    TH_air_loc      = (TH_air_anom_avg + T_0_nondim) * One2D  # absolute for MOST

    # Friction velocity (with floor to prevent near-zero division)
    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    # Diagnose surface heat flux — both TH_sfc_t and TH_air_anom_avg are anomalies
    # qz = -u* x th*,  th* = vonk x (TH_air - TH_s) / denom_h  (>0 for stable)
    denom_h    = jnp.log(0.5 * dz * z_scale / z0T) + psi2D_h - psi2D_h0
    qz_sfc_2D  = ustar * vonk * (TH_sfc_t - TH_air_anom_avg) * One2D / denom_h
    qz_sfc_avg = jnp.mean(qz_sfc_2D)

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_air_loc, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_2D, qz_sfc_avg, invOB, MOSTfunctions


@jax.jit
def SurfaceFlux_HeterogeneousPrescribedTemperature(u, v, TH, TH_sfc_t,
                                                    MOSTfunctions):
    """
    optSurfFlux=1, optSurfBC=2: heterogeneous surface, prescribed T_s(t).

    Uses local (per-column) wind speed and air temperature.
    TH_sfc_t is spatially uniform but temporally varying.

    Parameters:
    -----------
    TH_sfc_t : scalar JAX value, non-dimensional surface temperature anomaly
               (theta_sfc - T_0) / TH_scale at current timestep, as returned
               by Initialize_SurfaceBC for optSurfBC=2

    Returns:
    --------
    M_sfc_loc    : (nx, ny)
    ustar        : (nx, ny)
    qz_sfc_2D    : (nx, ny) diagnosed surface heat flux
    qz_sfc_avg   : scalar
    invOB        : (nx, ny)
    MOSTfunctions: updated tuple
    """

    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    # Local surface wind speed
    M_sfc_loc = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)

    # TH is stored as anomaly (TH - T_0); TH_sfc_t is also an anomaly.
    TH_air_anom_loc = TH[:, :, 0]

    # Friction velocity
    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar   = jnp.maximum(vonk * M_sfc_loc / denom_m, 1e-3)

    # Diagnose surface heat flux — both TH_sfc_t and TH_air_anom_loc are anomalies
    denom_h    = jnp.log(0.5 * dz * z_scale / z0T) + psi2D_h - psi2D_h0
    qz_sfc_2D  = ustar * vonk * (TH_sfc_t - TH_air_anom_loc) / denom_h
    qz_sfc_avg = jnp.mean(qz_sfc_2D)

    # Absolute TH_air as reference for Obukhov length
    TH_air_ref = (jnp.mean(TH_air_anom_loc) + T_0_nondim) * jnp.ones((nx, ny))

    MOSTfunctions, invOB = _update_MOSTfunctions(
        ustar, qz_sfc_avg, TH_air_ref, psi2D_m, psi2D_m0, psi2D_h, psi2D_h0)

    return M_sfc_loc, ustar, qz_sfc_2D, qz_sfc_avg, invOB, MOSTfunctions
