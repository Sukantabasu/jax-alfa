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
File: DerivedVars.py
==========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: Derived variables for JAXLES computed from base configuration
"""

# ============================================================
# Imports
# ============================================================

import jax.numpy as jnp
from .Config import *


# ============================================================
# Constants
# ============================================================

pi = jnp.pi
g = 9.81  # gravitational acceleration (m/s2)
vonk = 0.4  # von Karman constant


# ============================================================
# Derived Variables (computed from Config.py)
# ============================================================

# Velocity, potential temperature and length scales
u_scale = 1
TH_scale = 1
z_scale = l_x / (2 * pi)

# Non-dimensional g
g_nondim = g * z_scale / (u_scale ** 2)

# Non-dimensional Coriolis term
f_coriolis_nondim = f_coriolis * (z_scale / u_scale)

# Non-dimensional reference temperature
T_0_nondim = T_0 / TH_scale

# Non-dimensional inversion term
inversion_nondim = inversion * (z_scale / TH_scale)

# Non-dimensional surface sensible heat flux
qz_sfc = SensibleHeatFlux * jnp.ones((nx, ny)) / (u_scale * TH_scale)

# Non-dimensional grid spacing
dx = 2 * pi / nx
dy = 2 * pi / ny
dz = (l_z / z_scale) / (nz - 1)
idz = 1.0 / dz

# Filter width
L = FGR * (dx * dy * dz) ** (1 / 3)

# Non-dimensional time-step
dt_nondim = dt * u_scale / z_scale

# Total time steps
nsteps = int(np.ceil(SimTime / dt))

# Test filter ratio (TFR) & dealiasing option
if FGR == 1:
    TFR = 2
    optDealias = 1  # 0: no dealias, 1: dealias
else:
    TFR = np.sqrt(2)
    optDealias = 0  # 0: no dealias, 1: dealias

# Initialize Cs2 and Cs2PrRatio for static SGS models
Cs2_3D = Cs2 * jnp.ones((nx, ny, nz))
Cs2PrRatio_3D = Cs2PrRatio * jnp.ones((nx, ny, nz))

# Non-dimensional damping height
z_damping_nondim = z_damping / z_scale

# Relaxation time
RelaxTime = 600 * dt  # unit: s
RelaxTime_nondim = RelaxTime * (u_scale / z_scale)

# Collect samples every SampleInterval (steps)
SampleInterval = int( SampleInterval_sec / dt)
# Output averages every OutputInterval (steps)
OutputInterval = int( round(OutputInterval_sec / dt / 5) * 5)
# Output 3D fields every Output3DInterval (steps)
Output3DInterval = int( round(Output3DInterval_sec / dt / 5) * 5)
