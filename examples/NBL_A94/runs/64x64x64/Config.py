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
File: Config.py
===============

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-10-22
:Description: namelist file for JAX-ALFA runs
"""


# ============================================================
# Imports
# ============================================================

import numpy as np


# ============================================================
# User Input
# ============================================================

# ------------------------------------------------------------
# Platform options
# ------------------------------------------------------------
use_double_precision = True
# 0: use CPU, 1: use GPU
optGPU = 1
GPU_ID = 0

# ------------------------------------------------------------
# Domain configuration
# ------------------------------------------------------------

# Domain size (m)
l_x = 4000
l_y = 4000
l_z = 4000

# Number of grid points
nx = 64
ny = 64
nz = 64

# ------------------------------------------------------------
# Time integration configuration
# ------------------------------------------------------------

# Change this if it is a restart run
istep = 1

# Time stepping and simulation time
dt = 3  # unit: sec
SimTime = 300000  # unit: sec

# Galilean transformation
Ugal = 3  # unit: m/s

# ------------------------------------------------------------
# Surface configuration
# ------------------------------------------------------------

# 0: homogeneous, prescribed-flux
# 1: heterogeneous, prescribed-flux
optSurfFlux = 0

# Roughness lengths
z0m = 0.01  # unit: m
z0T = 0.01  # unit: m

SensibleHeatFlux = 0.0 # K m/s


# ------------------------------------------------------------
# Forcing configuration
# ------------------------------------------------------------

# Geostrophic wind speeds
Ug2 = 5  # unit: m/s
Vg2 = 0  # unit: m/s

# Coriolis term
f_coriolis = 1e-4  # unit: 1/s

# Inversion strength
inversion = 0/1000  # unit: K/m

# Buoyancy calculation
optBuoyancy = 1  #0: use reference T_0, 1: use local THv

T_0 = 300  # unit: K

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model to use
# 0: Static Smagorinsky
# 1: LASDD-SM (Locally-averaged Scale-dependent Dynamic, Smagorinsky)
# 2: LASDD-WL (Locally-averaged Scale-dependent Dynamic, Wong-Lilly)
# 3: LAD-SM  (Locally-averaged Dynamic, Smagorinsky; beta=1)
# 4: LAD-WL  (Locally-averaged Dynamic, Wong-Lilly; beta=1)
optSgs = 1

# How often dynamic SGS coefficients are computed?
dynamicSGS_call_time = 1

# Filter to grid ratio (recommended: FGR = 2)
# Dealiasing is not activated for FGR >= 2
FGR = 1  # FGR = 1 implies implicit filtering

# Initial SGS coefficients (used before first dynamic update when dynamicSGS_call_time > 1)
Cs2 = 0.1 ** 2        # SM models: initial Cs^2
Cwl = 0.1 ** 2        # WL models: initial C_WL
Cs2PrRatio = Cs2 / 1.0
CwlPrRatio = Cwl / 1.0

# ------------------------------------------------------------
# Damping layer configuration
# ------------------------------------------------------------

# Rayleigh damping to activate?
optDamping = 1  # 1: yes, 0: no

# Starting height of damping layer
z_damping = 2500  # unit: m

# Relaxation time
RelaxTime = 300  # unit: s

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

# Collect samples every SampleInterval_sec (sec)
SampleInterval_sec = 12.0 # ideally, should be divisible by dt
# Output averages every OutputInterval_sec (sec)
OutputInterval_sec = 60.0 # ideally, should be divisible by dt
# Output 3D fields every Output3DInterval_sec (sec)
Output3DInterval_sec = SimTime  # output 3D fields only at the end of the run

# ------------------------------------------------------------
# Surface BC configuration
# ------------------------------------------------------------
# 0: constant flux (SensibleHeatFlux), 1: time-varying flux, 2: time-varying surface T
optSurfBC = 0
SurfaceBCFile = 'input/SurfaceBC.npz'

# Screen-level temperature reference height (m); 0 = use z0T
zTemperature = 0.0

# ------------------------------------------------------------
# Geostrophic wind configuration
# ------------------------------------------------------------
# 0: constant (Ug2, Vg2 above), 1: time/height-varying (from GeoWindFile)
optGeoWind = 0
GeoWindFile = 'input/GeoWind.npz'

# ------------------------------------------------------------
# Large-scale advection forcing
# ------------------------------------------------------------
# 0: none, 1: time/height-varying (from AdvectionFile)
optAdvection = 0
AdvectionFile = 'input/AdvForcing.npz'

# ------------------------------------------------------------
# Moisture configuration
# ------------------------------------------------------------
# 0: dry run, 1: prognostic specific humidity Q
optMoisture = 0
# Screen-level moisture reference height (m); 0 = use z0T
zMoisture = 0.0
# Surface moisture flux (kg/kg m/s); used when optMoistureSurfBC = 0
MoistureFlux = 0.0
# 0: constant flux, 1: time-varying flux, 2: time-varying surface Q
optMoistureSurfBC = 0
MoistureSurfaceBCFile = 'input/MoistureSurfaceBC.npz'
# Specific humidity lapse rate above domain top (kg/kg/m); 0 = zero gradient
q_inversion = 0.0
