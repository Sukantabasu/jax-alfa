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
:Date: 2026-05-18
:Description: Namelist for the Wangara diurnal cycle case.
              Reference: Basu et al. (2008), Boundary-Layer Meteorology.
              Domain 5000x5000x2000 m, 80^3 grid, 24-hour simulation.
              Simulation period: full diurnal cycle from 0900 LST day 33
              (16 August 1967) to 0900 LST day 34 (17 August 1967).
              Surface BC: observed screen temperature at 1.2 m (zTemperature).
              Geostrophic wind: time- and height-varying (optGeoWind=1).
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
use_double_precision = False
# 0: use CPU, 1: use GPU
optGPU = 1
GPU_ID = 0

# ------------------------------------------------------------
# Domain configuration
# ------------------------------------------------------------

# Domain size (m)
l_x = 5000
l_y = 5000
l_z = 2000

# Number of grid points
nx = 80
ny = 80
nz = 80

# ------------------------------------------------------------
# Time integration configuration
# ------------------------------------------------------------

# Change this if it is a restart run
istep = 1

# Time stepping and simulation time
dt = 0.5          # unit: sec
SimTime = 86400   # 24 hours, unit: sec

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
z0T = 0.01   # not used when zTemperature > 0; kept for backward compatibility

# Screen-level temperature reference height (m).
# The Wangara surface temperature is measured at 1.2 m above the ground.
# MOST heat-flux denominator uses log(z1/zTemperature) + psi_h - psi_h0.
zTemperature = 1.2

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

# Constant geostrophic wind (m/s) — used only when optGeoWind = 0
Ug2 = -5.34
Vg2 = -0.43

# Path to geostrophic wind file (relative to run directory)
GeoWindFile = 'input/GeoWind.npz'

# Coriolis parameter: f = 2*Omega*sin(lat)
# Wangara, Australia (~34 deg S)  =>  f < 0
f_coriolis = -8.26e-5  # unit: 1/s

# Temperature inversion: 0 = no prescribed inversion (diurnal cycle case)
inversion = 0.0

# Buoyancy calculation: 1 = use fixed T_0
optBuoyancy = 2

# Reference temperature (K) — screen-level potential temperature at t=0
# theta_screen = (5.3 + 273.16) + 1.2*10/1000 = 278.47 K  (from Wangara_Sfc3309.txt)
T_0 = 278.5

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model: 1 = LASDD-SM, 2 = LASDD-WL, 3 = LAD-SM, 4 = LAD-WL
optSgs = 1

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

optDamping = 1       # 1: activate Rayleigh damping
z_damping  = 1500.0  # unit: m
RelaxTime  = 600.0   # unit: s

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

SampleInterval_sec   = 10.0       # collect a sample every 10 s
OutputInterval_sec   = 60.0       # output averaged stats every 60 s
Output3DInterval_sec = 24*3600.0  # output 3D fields at the end of the run
