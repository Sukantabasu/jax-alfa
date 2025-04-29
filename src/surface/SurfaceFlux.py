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
File: JL_SurfaceFlux.py
==============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: surface flux calculation module.
              Handles both homogeneous and heterogeneous boundary conditions,
              and both constant heat flux and prescribed surface temperature.
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *

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
#  Homogeneous with constant heat flux
# ============================================================

@jax.jit
def SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions):
    """
    Calculate surface fluxes for homogeneous boundary condition
    with constant sensible heat flux.

    Parameters:
    -----------
    u, v : jnp.ndarray of shape (nx, ny, nz)
        Velocity components
    TH : jnp.ndarray of shape (nx, ny, nz)
        Potential temperature
    SensibleHeatFlux : prescribed value in config.py
        Surface heat flux, unit: K m/s

    Returns:
    --------
    M_res : jnp.ndarray of shape (nx, ny)
        Mean wind speed near surface
    ustar : jnp.ndarray of shape (nx, ny)
        Friction velocity
    psi_m : jnp.ndarray of shape (nx, ny)
        Stability function for momentum
    psi_m0 : jnp.ndarray of shape (nx, ny)
        Surface stability function for momentum
    psi_h : jnp.ndarray of shape (nx, ny)
        Stability function for heat
    psi_h0 : jnp.ndarray of shape (nx, ny)
        Surface stability function for heat
    fi_m : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for momentum
    fi_m : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for heat
    SensibleHeatFlux : prescribed value in config.py
        Surface heat flux, unit: K m/s
    """

    One2D = jnp.ones((nx, ny))

    # unpack MOST functions
    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = jnp.mean(qz_sfc)

    # Compute grid average of wind speed at the lowest level
    M_sfc_avg = jnp.mean(jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2))
    M_sfc_loc = M_sfc_avg * One2D

    # Grid average of potential temperature at the lowest level
    TH_sfc_avg = jnp.mean(TH[:, :, 0])
    TH_sfc_loc = TH_sfc_avg * One2D

    # Compute ustar
    denom = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar = vonk * M_sfc_loc / denom

    # Compute inverse Obukhov length
    # OB = -(ustar ** 3) * TH_sfc_loc / (vonk * g_nondim * qz_sfc_avg)
    invOB = -(vonk * g_nondim * qz_sfc_avg) / ((ustar ** 3) * TH_sfc_loc)
    is_stable = qz_sfc_avg <= 0

    # Compute updated stability functions
    z1_over_L = (0.5 * dz) * invOB
    z0m_over_L = (z0m / z_scale) * invOB
    z0T_over_L = (z0T / z_scale) * invOB

    psi2D_m, psi2D_h, fi2D_m, fi2D_h = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z1_over_L),
        lambda _: MOSTunstable(z1_over_L),
        operand=None
    )

    psi2D_m0, _, fi2D_m0, _ = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0m_over_L),
        lambda _: MOSTunstable(z0m_over_L),
        operand=None
    )

    _, psi2D_h0, _, fi2D_h0 = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0T_over_L),
        lambda _: MOSTunstable(z0T_over_L),
        operand=None
    )

    MOSTfunctions = (psi2D_m, psi2D_m0, psi2D_h, psi2D_h0, fi2D_m, fi2D_h)

    return M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions


# ============================================================
#  Heterogeneous with constant heat flux
# ============================================================

@jax.jit
def SurfaceFlux_HeterogeneousConstantFlux(u, v, TH, MOSTfunctions):
    """
    Calculate surface fluxes for heterogeneous boundary condition
    with constant sensible heat flux.

    Parameters:
    -----------
    u, v : jnp.ndarray of shape (nx, ny, nz)
        Velocity components
    TH : jnp.ndarray of shape (nx, ny, nz)
        Potential temperature
    SensibleHeatFlux : prescribed value in config.py
        Surface heat flux, unit: K m/s

    Returns:
    --------
    M_sfc_loc : jnp.ndarray of shape (nx, ny)
        Mean wind speed near surface
    ustar : jnp.ndarray of shape (nx, ny)
        Friction velocity
    psi2D_m : jnp.ndarray of shape (nx, ny)
        Stability function for momentum
    psi2D_m0 : jnp.ndarray of shape (nx, ny)
        Surface stability function for momentum
    psi2D_h : jnp.ndarray of shape (nx, ny)
        Stability function for heat
    psi2D_h0 : jnp.ndarray of shape (nx, ny)
        Surface stability function for heat
    fi2D_m : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for momentum
    fi2D_m : jnp.ndarray of shape (nx, ny)
        Normalized gradient function for heat
    SensibleHeatFlux : prescribed value in config.py
        Surface heat flux, unit: K m/s
    """

    One2D = jnp.ones((nx, ny))

    # unpack MOST functions
    (psi2D_m, psi2D_m0,
     psi2D_h, psi2D_h0,
     fi2D_m, fi2D_h) = MOSTfunctions

    qz_sfc_avg = jnp.mean(qz_sfc)

    # Compute grid average of wind speed at the lowest level
    M_sfc_loc = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)

    # Grid average of potential temperature at the lowest level
    TH_sfc_loc = TH[:, :, 0]

    # Compute ustar
    denom = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar = vonk * M_sfc_loc / denom

    # Compute inverse Obukhov length
    # OB = -(ustar ** 3) * TH_sfc_loc / (vonk * g_nondim * qz_sfc_avg)
    invOB = -(vonk * g_nondim * qz_sfc_avg) / ((ustar ** 3) * TH_sfc_loc)
    is_stable = qz_sfc_avg <= 0

    # Compute updated stability functions
    z1_over_L = (0.5 * dz) * invOB
    z0m_over_L = (z0m / z_scale) * invOB
    z0T_over_L = (z0T / z_scale) * invOB

    psi2D_m, psi2D_h, fi2D_m, fi2D_h = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z1_over_L),
        lambda _: MOSTunstable(z1_over_L),
        operand=None
    )

    psi2D_m0, _, fi2D_m0, _ = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0m_over_L),
        lambda _: MOSTunstable(z0m_over_L),
        operand=None
    )

    _, psi2D_h0, _, fi2D_h0 = jax.lax.cond(
        is_stable,
        lambda _: MOSTstable(z0T_over_L),
        lambda _: MOSTunstable(z0T_over_L),
        operand=None
    )

    MOSTfunctions = (psi2D_m, psi2D_m0, psi2D_h, psi2D_h0, fi2D_m, fi2D_h)

    return M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions

