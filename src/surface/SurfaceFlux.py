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
:Date: 2025-4-5
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


# ============================================================
#  Homogeneous with prescribed near-surface air temperature
# ============================================================

@jax.jit
def SurfaceFlux_HomogeneousPrescribedTemperature(u, v, TH, MOSTfunctions):
    """
    Calculate surface fluxes for homogeneous boundary condition
    with prescribed near-surface air temperature.

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

# # ============================================================
# #  Unified surface flux calculation functions
# # ============================================================
#
# @jax.jit
# def surface_flux(u, v, T, homogeneous=True, const_heat=True, SHFX=None, t_s=None):
#     """
#
#     Parameters:
#     -----------
#     u, v : jnp.ndarray of shape (nx, ny, nz)
#         Velocity components
#     T : jnp.ndarray of shape (nx, ny, nz)
#         Potential temperature
#     homogeneous : bool
#         Whether to use homogeneous boundary conditions
#         (True for domain averages, False for local values)
#     const_heat : bool
#         Whether to use constant heat flux (True) or prescribed surface temperature (False)
#     SHFX : float or jnp.ndarray, optional
#         Surface heat flux, used if const_heat=True
#     t_s : float or jnp.ndarray, optional
#         Surface temperature, used if const_heat=False
#
#     Returns:
#     --------
#     M_res, ustar, fi, fiH, psi, psi0, psiH, psiH0, SHFX : see individual functions
#     """
#
#     # Call the appropriate function based on the flags
#     if homogeneous:
#         if const_heat:
#             return surface_flux_homogeneous_const_heat(u, v, T, SHFX)
#         else:
#             return surface_flux_homogeneous_surface_temp(u, v, T, t_s)
#     else:
#         if const_heat:
#             return surface_flux_heterogeneous_const_heat(u, v, T, SHFX)
#         else:
#             return surface_flux_heterogeneous_surface_temp(u, v, T, t_s)
#
#
# # ============================================================
# #  Utility functions for working with surface quantities
# # ============================================================
#
# @jax.jit
# def compute_surface_stress(M_res, ustar, u, v):
#     """
#     Computes the surface stress components from ustar.
#
#     Parameters:
#     -----------
#     M_res : jnp.ndarray of shape (nx, ny)
#         Wind speed at first level
#     ustar : jnp.ndarray of shape (nx, ny)
#         Friction velocity
#     u, v : jnp.ndarray of shape (nx, ny, nz)
#         Velocity components
#
#     Returns:
#     --------
#     txz_sfc, tyz_sfc : jnp.ndarray of shape (nx, ny)
#         Surface stress components
#     """
#     # Surface stress components computed using log law
#     txz_sfc = -(ustar ** 2) * (u[:, :, 0] + Ugal) / M_res
#     tyz_sfc = -(ustar ** 2) * v[:, :, 0] / M_res
#
#     return txz_sfc, tyz_sfc
#
#
# @jax.jit
# def compute_stability_parameters(OB, dz, z_i):
#     """
#     Computes various non-dimensional stability parameters for diagnostics.
#
#     Parameters:
#     -----------
#     OB : jnp.ndarray of shape (nx, ny)
#         Obukhov length
#     dz : float
#         Grid spacing in z-direction
#     z_i : float
#         Inversion height (or boundary layer height)
#
#     Returns:
#     --------
#     z_L : jnp.ndarray
#         Stability parameter at first grid level
#     z_L_inv : jnp.ndarray
#         Stability parameter for inversion height
#     non_dim_OB : jnp.ndarray
#         Non-dimensional Obukhov length
#     """
#     # Compute stability parameter at first grid level
#     z_L = (0.5 * dz) / OB
#
#     # Compute stability parameter for inversion height
#     z_L_inv = z_i / OB
#
#     # Non-dimensional Obukhov length
#     non_dim_OB = OB / z_i
#
#     return z_L, z_L_inv, non_dim_OB
#
#     z2_over_L = z2 / OB
#
#     # Get stability functions for level 2
#     psi_m_2, fi_m_2 = mo_stability_momentum(z2_over_L)
#     psi_h_2, fi_h_2 = mo_stability_heat(z2_over_L)
#
#     # Update arrays for level 2
#     psi = psi.at[:, :, 1].set(psi_m_2)
#     psi0 = psi0.at[:, :, 1].set(psi_m_0)  # Same surface value
#     fi = fi.at[:, :, 1].set(fi_m_2)
#     psiH = psiH.at[:, :, 1].set(psi_h_2)
#     psiH0 = psiH0.at[:, :, 1].set(psi_h_0)  # Same surface value
#     fiH = fiH.at[:, :, 1].set(fi_h_2)
#
#     # Second iteration: update ustar with stability functions
#     denom = jnp.log(0.5 * dz * z_i / z0) + psi[:, :, 0] - psi0[:, :, 0]
#     ustar = vonk * M_res / denom
#
#     return M_res, ustar, fi, fiH, psi, psi0, psiH, psiH0, SHFX
#
#
# # ============================================================
# #  Surface flux calculation - homogeneous with surface temperature
# # ============================================================
#
# @jax.jit
# def surface_flux_homogeneous_surface_temp(u, v, T, t_s=None):
#     """
#     Calculate surface fluxes for homogeneous boundary condition with prescribed surface temperature.
#
#     Parameters:
#     -----------
#     u, v : jnp.ndarray of shape (nx, ny, nz)
#         Velocity components
#     T : jnp.ndarray of shape (nx, ny, nz)
#         Potential temperature
#     t_s : float or jnp.ndarray, optional
#         Surface temperature; if None, defaults to T[:,:,0] + 1.0
#
#     Returns:
#     --------
#     M_res : jnp.ndarray of shape (nx, ny)
#         Mean wind speed near surface
#     ustar : jnp.ndarray of shape (nx, ny)
#         Friction velocity
#     fi : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for momentum
#     fiH : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for heat
#     psi : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for momentum
#     psi0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for momentum
#     psiH : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for heat
#     psiH0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for heat
#     SHFX : jnp.ndarray of shape (nx, ny)
#         Surface heat flux
#     """
#     nx, ny = u.shape[0:2]
#
#     # Set surface temperature if not provided
#     if t_s is None:
#         # Default to 1K above the first level temperature
#         t_s = jnp.mean(T[:, :, 0]) + 1.0
#
#     if isinstance(t_s, (int, float)) or (hasattr(t_s, 'shape') and t_s.shape == ()):
#         t_s_array = jnp.ones((nx, ny)) * t_s
#     else:
#         t_s_array = t_s
#
#     # Compute grid average of wind speed and temperature at the first level
#     M_avg = jnp.mean(jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2))
#     M_res = jnp.ones((nx, ny)) * M_avg
#
#     T_avg = jnp.mean(T[:, :, 0])
#     T_res = jnp.ones((nx, ny)) * T_avg
#
#     # Initialize arrays for stability functions
#     psi = jnp.ones((nx, ny, 2))
#     psi0 = jnp.ones((nx, ny, 2))
#     fi = jnp.ones((nx, ny, 2))
#     psiH = jnp.ones((nx, ny, 2))
#     psiH0 = jnp.ones((nx, ny, 2))
#     fiH = jnp.ones((nx, ny, 2))
#
#     # First iteration: compute ustar with neutral stability
#     denom = jnp.log(0.5 * dz * z_i / z0)
#     ustar = vonk * M_res / denom
#
#     # Initialize with neutral stability
#     denomH = jnp.log(0.5 * dz * z_i / z0T)
#
#     # Compute surface heat flux from surface temperature
#     SHFX = (t_s_array - T_res) * ustar * vonk / denomH
#     T_flux_avg = jnp.mean(SHFX)
#
#     # Compute Obukhov length
#     OB = -(ustar ** 3) * T_res / (vonk * g_nondim * T_flux_avg)
#
#     # Update stability functions for level 1 (k=0)
#     z1 = 0.5 * dz
#     z1_over_L = z1 / OB
#     z0_over_L = (z0 / z_i) / OB
#     z0T_over_L = (z0T / z_i) / OB
#
#     # Get stability functions for level 1
#     psi_m_1, fi_m_1 = mo_stability_momentum(z1_over_L)
#     psi_m_0, fi_m_0 = mo_stability_momentum(z0_over_L)
#     psi_h_1, fi_h_1 = mo_stability_heat(z1_over_L)
#     psi_h_0, fi_h_0 = mo_stability_heat(z0T_over_L)
#
#     # Update arrays for level 1
#     psi = psi.at[:, :, 0].set(psi_m_1)
#     psi0 = psi0.at[:, :, 0].set(psi_m_0)
#     fi = fi.at[:, :, 0].set(fi_m_1)
#     psiH = psiH.at[:, :, 0].set(psi_h_1)
#     psiH0 = psiH0.at[:, :, 0].set(psi_h_0)
#     fiH = fiH.at[:, :, 0].set(fi_h_1)
#
#     # Update stability functions for level 2 (k=1)
#     z2 = 1.0 * dz
#     z2_over_L = z2 / OB
#
#     # Get stability functions for level 2
#     psi_m_2, fi_m_2 = mo_stability_momentum(z2_over_L)
#     psi_h_2, fi_h_2 = mo_stability_heat(z2_over_L)
#
#     # Update arrays for level 2
#     psi = psi.at[:, :, 1].set(psi_m_2)
#     psi0 = psi0.at[:, :, 1].set(psi_m_0)  # Same surface value
#     fi = fi.at[:, :, 1].set(fi_m_2)
#     psiH = psiH.at[:, :, 1].set(psi_h_2)
#     psiH0 = psiH0.at[:, :, 1].set(psi_h_0)  # Same surface value
#     fiH = fiH.at[:, :, 1].set(fi_h_2)
#
#     # Second iteration: update ustar with stability functions
#     denom = jnp.log(0.5 * dz * z_i / z0) + psi[:, :, 0] - psi0[:, :, 0]
#     ustar = vonk * M_res / denom
#
#     # Update heat flux calculation with stability functions
#     denomH = jnp.log(0.5 * dz * z_i / z0T) + psiH[:, :, 0] - psiH0[:, :, 0]
#     SHFX = (t_s_array - T_res) * ustar * vonk / denomH
#
#     return M_res, ustar, fi, fiH, psi, psi0, psiH, psiH0, SHFX
#
#
# # ============================================================
# #  Surface flux calculation - heterogeneous with constant heat flux
# # ============================================================
#
# @jax.jit
# def surface_flux_heterogeneous_const_heat(u, v, T, SHFX=None):
#     """
#     Calculate surface fluxes for heterogeneous boundary condition with constant heat flux.
#
#     Parameters:
#     -----------
#     u, v : jnp.ndarray of shape (nx, ny, nz)
#         Velocity components
#     T : jnp.ndarray of shape (nx, ny, nz)
#         Potential temperature
#     SHFX : float or jnp.ndarray, optional
#         Prescribed heat flux; if None, uses t_flux from Config
#
#     Returns:
#     --------
#     M_res : jnp.ndarray of shape (nx, ny)
#         Actual wind speed near surface at each point
#     ustar : jnp.ndarray of shape (nx, ny)
#         Friction velocity at each point
#     fi : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for momentum
#     fiH : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for heat
#     psi : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for momentum
#     psi0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for momentum
#     psiH : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for heat
#     psiH0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for heat
#     SHFX : jnp.ndarray of shape (nx, ny)
#         Surface heat flux
#     """
#     nx, ny = u.shape[0:2]
#
#     # Use default heat flux if not provided
#     if SHFX is None:
#         SHFX = jnp.ones((nx, ny)) * t_flux
#     elif isinstance(SHFX, (int, float)) or (hasattr(SHFX, 'shape') and SHFX.shape == ()):
#         SHFX = jnp.ones((nx, ny)) * SHFX
#
#     # Compute wind speed at each point in the first level
#     M_res = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)
#
#     # Temperature at each point in the first level
#     T_res = T[:, :, 0]
#
#     # Surface heat flux
#     T_flux_avg = jnp.mean(SHFX)
#
#     # Initialize arrays for stability functions
#     psi = jnp.ones((nx, ny, 2))
#     psi0 = jnp.ones((nx, ny, 2))
#     fi = jnp.ones((nx, ny, 2))
#     psiH = jnp.ones((nx, ny, 2))
#     psiH0 = jnp.ones((nx, ny, 2))
#     fiH = jnp.ones((nx, ny, 2))
#
#     # First iteration: compute ustar with neutral stability
#     denom = jnp.log(0.5 * dz * z_i / z0)
#     ustar = vonk * M_res / denom
#
#     # Compute Obukhov length (with small value protection for division)
#     denominator = vonk * g_nondim * T_flux_avg
#     OB = jnp.where(jnp.abs(denominator) > 1e-10,
#                    -(ustar ** 3) * T_res / denominator,
#                    -1000.0)  # Default large negative value for near-neutral conditions
#
#     # Update stability functions for level 1 (k=0)
#     z1 = 0.5 * dz
#     z1_over_L = z1 / OB
#     z0_over_L = (z0 / z_i) / OB
#     z0T_over_L = (z0T / z_i) / OB
#
#     # Get stability functions for level 1
#     psi_m_1, fi_m_1 = mo_stability_momentum(z1_over_L)
#     psi_m_0, fi_m_0 = mo_stability_momentum(z0_over_L)
#     psi_h_1, fi_h_1 = mo_stability_heat(z1_over_L)
#     psi_h_0, fi_h_0 = mo_stability_heat(z0T_over_L)
#
#     # Update arrays for level 1
#     psi = psi.at[:, :, 0].set(psi_m_1)
#     psi0 = psi0.at[:, :, 0].set(psi_m_0)
#     fi = fi.at[:, :, 0].set(fi_m_1)
#     psiH = psiH.at[:, :, 0].set(psi_h_1)
#     psiH0 = psiH0.at[:, :, 0].set(psi_h_0)
#     fiH = fiH.at[:, :, 0].set(fi_h_1)
#
#     # Update stability functions for level 2 (k=1)
#     z2 = 1.0 * dz
#     z2_over_L = z2 / OB
#
#     # Get stability functions for level 2
#     psi_m_2, fi_m_2 = mo_stability_momentum(z2_over_L)
#     psi_h_2, fi_h_2 = mo_stability_heat(z2_over_L)
#
#     # Update arrays for level 2
#     psi = psi.at[:, :, 1].set(psi_m_2)
#     psi0 = psi0.at[:, :, 1].set(psi_m_0)  # Same surface value
#     fi = fi.at[:, :, 1].set(fi_m_2)
#     psiH = psiH.at[:, :, 1].set(psi_h_2)
#     psiH0 = psiH0.at[:, :, 1].set(psi_h_0)  # Same surface value
#     fiH = fiH.at[:, :, 1].set(fi_h_2)
#
#     # Second iteration: update ustar with stability functions
#     denom = jnp.log(0.5 * dz * z_i / z0) + psi[:, :, 0] - psi0[:, :, 0]
#     ustar = vonk * M_res / denom
#
#     return M_res, ustar, fi, fiH, psi, psi0, psiH, psiH0, SHFX
#
#
# # ============================================================
# #  Surface flux calculation - heterogeneous with surface temperature
# # ============================================================
#
# @jax.jit
# def surface_flux_heterogeneous_surface_temp(u, v, T, t_s=None):
#     """
#     Calculate surface fluxes for heterogeneous boundary condition with prescribed surface temperature.
#
#     Parameters:
#     -----------
#     u, v : jnp.ndarray of shape (nx, ny, nz)
#         Velocity components
#     T : jnp.ndarray of shape (nx, ny, nz)
#         Potential temperature
#     t_s : float or jnp.ndarray, optional
#         Surface temperature; if None, defaults to T[:,:,0] + 1.0
#
#     Returns:
#     --------
#     M_res : jnp.ndarray of shape (nx, ny)
#         Actual wind speed near surface at each point
#     ustar : jnp.ndarray of shape (nx, ny)
#         Friction velocity at each point
#     fi : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for momentum
#     fiH : jnp.ndarray of shape (nx, ny, 2)
#         Normalized gradient function for heat
#     psi : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for momentum
#     psi0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for momentum
#     psiH : jnp.ndarray of shape (nx, ny, 2)
#         Stability function for heat
#     psiH0 : jnp.ndarray of shape (nx, ny, 2)
#         Surface stability function for heat
#     SHFX : jnp.ndarray of shape (nx, ny)
#         Surface heat flux
#     """
#     nx, ny = u.shape[0:2]
#
#     # Set surface temperature if not provided
#     if t_s is None:
#         # Default to 1K above the local first level temperature
#         t_s_array = T[:, :, 0] + 1.0
#     elif isinstance(t_s, (int, float)) or (hasattr(t_s, 'shape') and t_s.shape == ()):
#         t_s_array = jnp.ones((nx, ny)) * t_s
#     else:
#         t_s_array = t_s
#
#     # Compute wind speed at each point in the first level
#     M_res = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)
#
#     # Temperature at each point in the first level
#     T_res = T[:, :, 0]
#
#     # Initialize arrays for stability functions
#     psi = jnp.ones((nx, ny, 2))
#     psi0 = jnp.ones((nx, ny, 2))
#     fi = jnp.ones((nx, ny, 2))
#     psiH = jnp.ones((nx, ny, 2))
#     psiH0 = jnp.ones((nx, ny, 2))
#     fiH = jnp.ones((nx, ny, 2))
#
#     # First iteration: compute ustar with neutral stability
#     denom = jnp.log(0.5 * dz * z_i / z0)
#     ustar = vonk * M_res / denom
#
#     # Initialize with neutral stability
#     denomH = jnp.log(0.5 * dz * z_i / z0T)
#
#     # Compute surface heat flux from surface temperature
#     SHFX = (t_s_array - T_res) * ustar * vonk / denomH
#     T_flux_avg = jnp.mean(SHFX)
#
#     # Compute Obukhov length (with protection for division by small values)
#     denominator = vonk * g_nondim * T_flux_avg
#     OB = jnp.where(jnp.abs(denominator) > 1e-10,
#                    -(ustar ** 3) * T_res / denominator,
#                    -1000.0)  # Default large negative value for near-neutral conditions
#
#     # Update stability functions for level 1 (k=0)
#     z1 = 0.5 * dz
#     z1_over_L = z1 / OB
#     z0_over_L = (z0 / z_i) / OB
#     z0T_over_L = (z0T / z_i) / OB
#
#     # Get stability functions for level 1
#     psi_m_1, fi_m_1 = mo_stability_momentum(z1_over_L)
#     psi_m_0, fi_m_0 = mo_stability_momentum(z0_over_L)
#     psi_h_1, fi_h_1 = mo_stability_heat(z1_over_L)
#     psi_h_0, fi_h_0 = mo_stability_heat(z0T_over_L)
#
#     # Update arrays for level 1
#     psi = psi.at[:, :, 0].set(psi_m_1)
#     psi0 = psi0.at[:, :, 0].set(psi_m_0)
#     fi = fi.at[:, :, 0].set(fi_m_1)
#     psiH = psiH.at[:, :, 0].set(psi_h_1)
#     psiH0 = psiH0.at[:, :, 0].set(psi_h_0)
#     fiH = fiH.at[:, :, 0].set(fi_h_1)
#
#     # Update stability functions for level 2 (k=1)
#     z2 = 1.0 * dz
