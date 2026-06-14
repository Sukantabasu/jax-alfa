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
:Date: 2026-06-14
:Description: GABLS4 stable BL at Dome C, Antarctica (Stage 3).
              Domain 400x400x1000 m; 36-hour simulation.
              Grid: 64x64x64, SGS: LASDD-SM,
              Precision: double.
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
l_x = 400
l_y = 400
l_z = 1000

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
dt = 0.1          # unit: sec
SimTime = 36 * 3600   # unit: sec

# Galilean transformation (m/s)
Ugal = 0

# ------------------------------------------------------------
# Surface configuration
# ------------------------------------------------------------

# optSurfFlux: 0 = homogeneous, 1 = heterogeneous
optSurfFlux = 0

# optSurfBC: 0 = constant flux
#            1 = time-varying flux (from SurfaceBCFile)
#            2 = time-varying surface temperature (from SurfaceBCFile)
optSurfBC = 2

# Roughness lengths (m)
z0m = 0.01
z0T = 0.01

# Screen-level temperature reference height (m); 0 = use z0T
zTemperature = 0.0

# SensibleHeatFlux: not used when optSurfBC >= 1; set to 0 as placeholder
SensibleHeatFlux = 0.0  # K m/s

# Path to surface BC file (relative to run directory)
SurfaceBCFile = 'input/SurfaceBC.npz'

# ------------------------------------------------------------
# Forcing configuration
# ------------------------------------------------------------

# Geostrophic wind option:
#   0 = constant Ug2, Vg2 from Config
#   1 = time + height varying, loaded from GeoWindFile
optGeoWind = 1

# Near-surface geostrophic wind at t=0 (m/s); fallback when optGeoWind=0
Ug2 = 1.25
Vg2 = 4.5

# Path to geostrophic wind file (relative to run directory)
GeoWindFile = 'input/GeoWind.npz'

# Coriolis parameter (1/s); Dome C at 75.1 deg S
f_coriolis = 1.409e-4

# Potential temperature lapse rate above domain top (K/m)
inversion = 0.0078

# Buoyancy calculation: 0 = use reference T_0, 1 = use local THv
optBuoyancy = 1

# Reference temperature (K)
T_0 = 271.0

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model: 0 = Static SM, 1 = LASDD-SM, 2 = LASDD-WL, 3 = LAD-SM, 4 = LAD-WL
optSgs = 1

# Dynamic SGS update frequency (every N steps)
dynamicSGS_call_time = 1

# Filter to grid ratio (FGR=1: implicit + dealiasing; FGR>=2: explicit)
FGR = 2

# Initial SGS coefficients (used before first dynamic update)
Cs2 = 0.1 ** 2        # SM models: initial Cs^2
Cwl = 0.1 ** 2        # WL models: initial C_WL
Cs2PrRatio = Cs2 / 1.0
CwlPrRatio = Cwl / 1.0

# ------------------------------------------------------------
# Damping layer configuration
# ------------------------------------------------------------

optDamping = 1       # 1: activate Rayleigh damping
z_damping  = 700    # unit: m
RelaxTime  = 60     # unit: s

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

SampleInterval_sec   = 10.0   # collect a sample every N s
OutputInterval_sec   = 600.0  # output averaged stats every N s
Output3DInterval_sec = SimTime   # output 3D fields only at the end of the run

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

# Pressure solver: 0 = LU (original), 1 = Thomas (tridiagonal, faster)
optPressureSolver = 1
