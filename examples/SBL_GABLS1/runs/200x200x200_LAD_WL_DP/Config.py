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
=======================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-08
:Description: namelist for the GABLS1 stable boundary layer benchmark.
              Reference: Beare et al. (2006), Boundary-Layer Meteorology.
              Domain 400x400x400 m, 200^3 grid, surface cooling 0.25 K/hr.
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
l_z = 400

# Number of grid points
nx = 200
ny = 200
nz = 200

# ------------------------------------------------------------
# Time integration configuration
# ------------------------------------------------------------

# Change this if it is a restart run
istep = 1

# Time stepping and simulation time
dt = 0.05       # unit: sec
SimTime = 9 * 3600  # 9 hours, unit: sec

# Galilean transformation
Ugal = 5  # unit: m/s

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
z0m = 0.1
z0T = 0.1

# SensibleHeatFlux: not used when optSurfBC >= 1; set to 0 as placeholder
SensibleHeatFlux = 0.0  # K m/s

# Path to surface BC file (relative to run directory)
SurfaceBCFile = 'input/SurfaceBC.npz'

# ------------------------------------------------------------
# Forcing configuration
# ------------------------------------------------------------

# Geostrophic wind speeds (m/s)
Ug2 = 8.0
Vg2 = 0.0

# Coriolis parameter: f = 2*Omega*sin(lat), lat=73 deg N
f_coriolis = 1.39e-4  # unit: 1/s

# Temperature inversion above 100 m (K/m)
inversion = 0.01

# Buoyancy calculation: 1 = use fixed T_0
optBuoyancy = 1

# Reference temperature (K) — initial surface temperature
T_0 = 265.0

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model: 0 = Static SM, 1 = LASDD-SM, 2 = LASDD-WL, 3 = LAD-SM, 4 = LAD-WL
optSgs = 4

# Dynamic SGS update frequency (every N steps)
dynamicSGS_call_time = 1

# Filter to grid ratio (FGR=1: implicit filtering with dealiasing)
FGR = 2

# Initial SGS coefficients (used before first dynamic update when dynamicSGS_call_time > 1)
Cs2 = 0.1 ** 2        # SM models: initial Cs^2
Cwl = 0.1 ** 2        # WL models: initial C_WL
Cs2PrRatio = Cs2 / 1.0
CwlPrRatio = Cwl / 1.0

# ------------------------------------------------------------
# Damping layer configuration
# ------------------------------------------------------------

optDamping = 1      # 1: activate Rayleigh damping
z_damping  = 300.0  # unit: m
RelaxTime  = 60.0   # unit: s

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

SampleInterval_sec   = 10.0    # collect a sample every 10 s
OutputInterval_sec   = 60.0    # output averaged stats every 1 min
Output3DInterval_sec = 9*3600.0  # output 3D fields every 1 hour
