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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
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
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *

# Import StagGridAvg
from ..utilities.Utilities import StagGridAvg

# dumDir is identified based on the location of Initialization.py
dumDir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
InputDir = os.path.join(dumDir, 'input')


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

    InputVelocity = os.path.join(InputDir, 'vel.ini')
    vel = np.loadtxt(InputVelocity)
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

    InputTH = os.path.join(InputDir, 'TH.ini')
    TH = np.loadtxt(InputTH)

    TH = np.reshape(TH, (nx, ny, nz), order='F')

    return jnp.array(TH)


# ============================================================
# Geostrophic wind components
# ============================================================

def Initialize_GeoWind():
    """
    Returns:
    --------
    Ug, Vg : ndarray
        3D arrays of size (nx, ny, nz) containing the initialized
        geostrophic wind components
    """

    Ug = Ug2*jnp.ones((nx, ny, nz))/u_scale
    Vg = Vg2*jnp.ones((nx, ny, nz))/u_scale

    return Ug, Vg


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
