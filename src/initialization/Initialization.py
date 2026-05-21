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
File: Initialization.py
==============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: loads velocity and temperature fields & reshape them
"""


# ============================================================
#  Imports
# ============================================================

import os
import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *

# Import StagGridAvg
from ..utilities.Utilities import StagGridAvg

InputDir = os.path.join(os.environ['JAXALFA_RUNDIR'], 'input')

# Absolute paths resolved once at import time
_SurfaceBCFile        = os.path.join(os.environ['JAXALFA_RUNDIR'], SurfaceBCFile)
_GeoWindFile          = os.path.join(os.environ['JAXALFA_RUNDIR'], GeoWindFile)
_AdvectionFile        = os.path.join(os.environ['JAXALFA_RUNDIR'], AdvectionFile)
_MoistureSurfaceBCFile = os.path.join(os.environ['JAXALFA_RUNDIR'], MoistureSurfaceBCFile)


# ============================================================
# Load velocity field
# ============================================================

def Initialize_uvw():
    """
    Returns:
    --------
    u, v, w : ndarray
        3D arrays of size (nx, ny, nz) containing the initialized
        velocity components
    """

    InputVelocity = os.path.join(InputDir, 'vel.npy')
    vel = np.load(InputVelocity)
    u = vel[:, 0] - Ugal
    v = vel[:, 1]
    w = vel[:, 2]

    u = np.reshape(u, (nx, ny, nz), order='F') / u_scale
    v = np.reshape(v, (nx, ny, nz), order='F') / u_scale
    w = np.reshape(w, (nx, ny, nz), order='F') / u_scale

    return jnp.array(u), jnp.array(v), jnp.array(w)


# ============================================================
# Load potential temperature field
# ============================================================

def Initialize_TH():
    """
    Returns:
    --------
    TH : ndarray
        3D array of size (nx, ny, nz) containing the initialized
        potential temperature field
    """

    InputTH = os.path.join(InputDir, 'TH.npy')
    TH = np.load(InputTH)

    TH = np.reshape(TH, (nx, ny, nz), order='F')

    # Subtract base state in numpy float64 before JAX cast; TH is stored as
    # anomaly TH' = TH - T_0 throughout the simulation.
    return jnp.array(TH - T_0_nondim)


# ============================================================
# Geostrophic wind components
# ============================================================

def Initialize_SurfaceBC():
    """
    Load the time-varying surface BC series (heat flux or surface temperature)
    from SurfaceBCFile. Called once before the main loop when optSurfBC >= 1.

    Validates that the file's dt_surf matches Config dt, and that the series
    length matches nsteps+1.

    Returns:
    --------
    SurfaceBC_series : jnp.ndarray of shape (nsteps+1,)
        Non-dimensional surface BC values at every timestep from t=0 to t=SimTime.
        For optSurfBC=1: non-dim heat flux     = data / (u_scale * TH_scale)
        For optSurfBC=2: non-dim temp anomaly  = (data - T_0) / TH_scale
            Stored as anomaly so float32 represents small values (~0 to -2.25 K)
            rather than absolute temperature (~265 K). Surface flux functions
            expect this anomaly and compare it against TH_air anomalies.
    """

    data = np.load(_SurfaceBCFile)

    # --- Validation ---
    dt_surf_file = float(data['dt_surf'])
    if abs(dt_surf_file - dt) > 1e-6:
        raise ValueError(
            f"SurfaceBC file dt_surf={dt_surf_file:.6f} s does not match "
            f"Config dt={dt:.6f} s. Re-run CreateSurfaceBC with the current Config."
        )

    optSurfBC_file = int(data['optSurfBC'])
    if optSurfBC_file != optSurfBC:
        raise ValueError(
            f"SurfaceBC file optSurfBC={optSurfBC_file} does not match "
            f"Config optSurfBC={optSurfBC}. Re-run CreateSurfaceBC."
        )

    series = data['data_series']
    expected_len = nsteps + 1
    if len(series) != expected_len:
        raise ValueError(
            f"SurfaceBC series length {len(series)} != nsteps+1={expected_len}. "
            f"Re-run CreateSurfaceBC with the current Config."
        )

    # --- Non-dimensionalise ---
    # optSurfBC=2: store as anomaly (theta_sfc - T_0) / TH_scale.
    # Subtraction happens here in NumPy float64, so the small differences
    # (0 to -2.25 K) are stored accurately when JAX casts to float32.
    if optSurfBC == 1:
        series_nondim = series / (u_scale * TH_scale)
    else:
        series_nondim = (series - T_0) / TH_scale

    return jnp.array(series_nondim)


def Initialize_GeoWind():
    """
    Returns constant (nx, ny, nz) geostrophic wind arrays from Config scalars
    Ug2, Vg2.  Used when optGeoWind == 0.
    """
    Ug = Ug2 * jnp.ones((nx, ny, nz)) / u_scale
    Vg = Vg2 * jnp.ones((nx, ny, nz)) / u_scale
    return Ug, Vg


def Initialize_GeoWind_Varying():
    """
    Load the time- and height-varying geostrophic wind series from GeoWindFile.
    Called once before the main loop when optGeoWind >= 1.

    Validates that the file's dt_geo matches Config dt and that the series
    dimensions match nsteps+1 and nz.

    Returns:
    --------
    GeoWind_U, GeoWind_V : jnp.ndarray of shape (nsteps+1, nz)
        Non-dimensional geostrophic wind profiles at every timestep.
        Index [iteration-1] gives the profile for that iteration.
    """

    data = np.load(_GeoWindFile)

    dt_geo_file = float(data['dt_geo'])
    if abs(dt_geo_file - dt) > 1e-6:
        raise ValueError(
            f"GeoWind file dt_geo={dt_geo_file:.6f} s does not match "
            f"Config dt={dt:.6f} s. Re-run CreateGeoWind with the current Config."
        )

    optGeoWind_file = int(data['optGeoWind'])
    if optGeoWind_file != optGeoWind:
        raise ValueError(
            f"GeoWind file optGeoWind={optGeoWind_file} does not match "
            f"Config optGeoWind={optGeoWind}. Re-run CreateGeoWind."
        )

    if 'Ug_series' in data and 'Vg_series' in data:
        Ug_series = data['Ug_series']   # (nsteps+1, nz) dimensional m/s
        Vg_series = data['Vg_series']   # (nsteps+1, nz) dimensional m/s

        if Ug_series.shape[0] != nsteps + 1:
            raise ValueError(
                f"GeoWind series length {Ug_series.shape[0]} != nsteps+1={nsteps + 1}. "
                f"Re-run CreateGeoWind with the current Config."
            )
        if Ug_series.shape[1] != nz:
            raise ValueError(
                f"GeoWind nz={Ug_series.shape[1]} != Config nz={nz}. "
                f"Re-run CreateGeoWind with the current Config."
            )
    else:
        t_profile = data['t_profile']       # (ntimes,) seconds
        Ug_profile = data['Ug_profile']     # (ntimes, nz) dimensional m/s
        Vg_profile = data['Vg_profile']     # (ntimes, nz) dimensional m/s

        if Ug_profile.shape[1] != nz:
            raise ValueError(
                f"GeoWind nz={Ug_profile.shape[1]} != Config nz={nz}. "
                f"Re-run CreateGeoWind with the current Config."
            )

        t_target = np.arange(nsteps + 1) * dt
        if t_target[0] < t_profile[0] - 1e-6 or t_target[-1] > t_profile[-1] + 1e-6:
            raise ValueError(
                "GeoWind compact profiles do not cover the configured simulation "
                f"interval 0-{t_target[-1]:.1f} s. Re-run CreateGeoWind."
            )

        Ug_series = np.zeros((nsteps + 1, nz))
        Vg_series = np.zeros((nsteps + 1, nz))
        for k in range(nz):
            Ug_series[:, k] = np.interp(t_target, t_profile, Ug_profile[:, k])
            Vg_series[:, k] = np.interp(t_target, t_profile, Vg_profile[:, k])

    # Non-dimensionalise (u_scale = 1 in current JAX-ALFA, kept for generality)
    GeoWind_U = jnp.array(Ug_series / u_scale)
    GeoWind_V = jnp.array(Vg_series / u_scale)

    return GeoWind_U, GeoWind_V


# ============================================================
# Large-scale (mesoscale) advection forcing
# ============================================================

def Initialize_AdvForcing():
    """
    Load the time- and height-varying large-scale advection forcing from
    AdvectionFile.  Called once before the main loop when optAdvection >= 1.

    The .npz file must contain:
        Uadv_series  : (nsteps+1, nz)  dimensional  [m s⁻²]
        Vadv_series  : (nsteps+1, nz)  dimensional  [m s⁻²]
        THadv_series : (nsteps+1, nz)  dimensional  [K s⁻¹]   (optional)
        Qadv_series  : (nsteps+1, nz)  dimensional  [kg/kg s⁻¹] (optional)
        dt_adv       : float           [s]  — must equal Config dt
        optAdvection : int32

    Returns:
    --------
    AdvForcing_U, AdvForcing_V, AdvForcing_TH, AdvForcing_Q :
        jnp.ndarray of shape (nsteps+1, nz)
        Non-dimensional advection tendencies at every timestep.
        Missing series default to zero.
        Non-dimensionalisation: [m s⁻²] × z_scale  (u_scale = TH_scale = Q_scale = 1).
    """

    data = np.load(_AdvectionFile)

    dt_adv_file = float(data['dt_adv'])
    if abs(dt_adv_file - dt) > 1e-6:
        raise ValueError(
            f"AdvForcing file dt_adv={dt_adv_file:.6f} s does not match "
            f"Config dt={dt:.6f} s. Re-run CreateAdvForcing with the current Config."
        )

    optAdvection_file = int(data['optAdvection'])
    if optAdvection_file != optAdvection:
        raise ValueError(
            f"AdvForcing file optAdvection={optAdvection_file} does not match "
            f"Config optAdvection={optAdvection}. Re-run CreateAdvForcing."
        )

    Uadv_series = data['Uadv_series']   # (nsteps+1, nz)  [m/s^2]
    Vadv_series = data['Vadv_series']   # (nsteps+1, nz)  [m/s^2]

    if Uadv_series.shape[0] != nsteps + 1:
        raise ValueError(
            f"AdvForcing series length {Uadv_series.shape[0]} != nsteps+1={nsteps + 1}. "
            f"Re-run CreateAdvForcing with the current Config."
        )
    if Uadv_series.shape[1] != nz:
        raise ValueError(
            f"AdvForcing nz={Uadv_series.shape[1]} != Config nz={nz}. "
            f"Re-run CreateAdvForcing with the current Config."
        )

    # Nondimensionalise: [m/s^2] * z_scale / u_scale^2  (u_scale = 1)
    AdvForcing_U  = jnp.array(Uadv_series * z_scale / u_scale ** 2)
    AdvForcing_V  = jnp.array(Vadv_series * z_scale / u_scale ** 2)

    if 'THadv_series' in data:
        THadv_series = data['THadv_series']   # (nsteps+1, nz)  [K/s]
        # Nondimensionalise: [K/s] * z_scale / (u_scale * TH_scale)  (both = 1)
        AdvForcing_TH = jnp.array(THadv_series * z_scale / (u_scale * TH_scale))
    else:
        AdvForcing_TH = jnp.zeros((nsteps + 1, nz))

    if 'Qadv_series' in data:
        Qadv_series = data['Qadv_series']   # (nsteps+1, nz)  [kg/kg/s]
        # Nondimensionalise: [kg/kg/s] * z_scale / (u_scale * Q_scale)  (both = 1)
        AdvForcing_Q = jnp.array(Qadv_series * z_scale / (u_scale * Q_scale))
    else:
        AdvForcing_Q = jnp.zeros((nsteps + 1, nz))

    return AdvForcing_U, AdvForcing_V, AdvForcing_TH, AdvForcing_Q


# ============================================================
# Moisture field
# ============================================================

def Initialize_Q():
    """
    Load the initial specific humidity field from input/Q.ini.
    Q is stored as absolute values (kg/kg) — no base-state subtraction.

    Returns:
    --------
    Q : jnp.ndarray of shape (nx, ny, nz)  [kg/kg]
    """
    InputQ = os.path.join(InputDir, 'Q.npy')
    Q = np.load(InputQ)
    Q = np.reshape(Q, (nx, ny, nz), order='F')
    return jnp.array(Q)


def Initialize_MoistureSurfaceBC():
    """
    Load the time-varying moisture surface BC series from MoistureSurfaceBCFile.
    Called once before the main loop when optMoisture=1 and optMoistureSurfBC >= 1.

    Returns:
    --------
    MoistureSurfaceBC_series : jnp.ndarray of shape (nsteps+1,)
        Non-dimensional values at every timestep.
        For optMoistureSurfBC=1: non-dim moisture flux = data / (u_scale * Q_scale)
        For optMoistureSurfBC=2: surface Q in kg/kg (stored as-is, already dimensional)
    """
    data = np.load(_MoistureSurfaceBCFile)

    dt_moist_file = float(data['dt_moist'])
    if abs(dt_moist_file - dt) > 1e-6:
        raise ValueError(
            f"MoistureSurfaceBC file dt_moist={dt_moist_file:.6f} s does not match "
            f"Config dt={dt:.6f} s. Re-run CreateMoistureSurfaceBC with the current Config."
        )

    optMoistureSurfBC_file = int(data['optMoistureSurfBC'])
    if optMoistureSurfBC_file != optMoistureSurfBC:
        raise ValueError(
            f"MoistureSurfaceBC file optMoistureSurfBC={optMoistureSurfBC_file} does not match "
            f"Config optMoistureSurfBC={optMoistureSurfBC}. Re-run CreateMoistureSurfaceBC."
        )

    series = data['data_series']
    expected_len = nsteps + 1
    if len(series) != expected_len:
        raise ValueError(
            f"MoistureSurfaceBC series length {len(series)} != nsteps+1={expected_len}. "
            f"Re-run CreateMoistureSurfaceBC with the current Config."
        )

    if optMoistureSurfBC == 1:
        # Flux in kg/kg m/s; non-dimensionalise by u_scale * Q_scale
        series_nondim = series / (u_scale * Q_scale)
    else:
        # optMoistureSurfBC == 2: surface Q (kg/kg); store as-is
        series_nondim = series

    return jnp.array(series_nondim)


# ============================================================
# Rayleigh damping layer
# ============================================================

def Initialize_RayleighDampingLayer():
    """
    Returns:
    --------
    RayleighDampCoeff : jnp.ndarray
        3D array of size (nx, ny, nz) containing the initialized
        Rayleigh damping layer coefficients
    """

    # Inverse non-dimensional relaxation time
    invRelaxTime_nondim = 1.0 / RelaxTime_nondim

    # Valid for both full and half levels
    z_top_nondim = l_z / z_scale

    # Calculate the damping layer depth
    RayleighDampThickness = z_top_nondim - z_damping_nondim

    #--------------------------------------------
    # Full levels
    #--------------------------------------------

    # Generate height levels
    z_nondim = jnp.arange(nz) * dz

    # Create mask for damping region
    RayleighDampMask = ((z_nondim >= z_damping_nondim) &
                        (z_nondim <= z_top_nondim))

    # Compute damping coefficient where mask is True
    RayleighDampCoeff1D = jnp.where(
        RayleighDampMask,
        0.5 * invRelaxTime_nondim * (1.0 - jnp.cos(
            jnp.pi * (z_nondim - z_damping_nondim) / RayleighDampThickness)),
        0.0)

    # Broadcast to 3D array
    RayleighDampCoeff = jnp.broadcast_to(
        RayleighDampCoeff1D.reshape(1, 1, nz),
        (nx, ny, nz))

    #--------------------------------------------
    # Half levels
    #--------------------------------------------

    # Generate height levels
    z_stag_nondim = (jnp.arange(nz) + 0.5) * dz

    # Create mask for damping region
    RayleighDampMask_stag = ((z_stag_nondim >= z_damping_nondim) &
                             (z_stag_nondim <= z_top_nondim))

    # Compute damping coefficient where mask is True
    RayleighDampCoeff1D_stag = jnp.where(
        RayleighDampMask_stag,
        0.5 * invRelaxTime_nondim * (1.0 - jnp.cos(
            jnp.pi * (z_stag_nondim - z_damping_nondim) /
            RayleighDampThickness)),
        0.0)

    # Broadcast to 3D array
    RayleighDampCoeff_stag = jnp.broadcast_to(
        RayleighDampCoeff1D_stag.reshape(1, 1, nz),
        (nx, ny, nz))

    return RayleighDampCoeff, RayleighDampCoeff_stag
