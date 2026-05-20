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
:Date: 2026-05-19
:Description: Namelist for the GABLS3 intercomparison case.
              Reference: Basu (2008), GABLS3 LES Intercomparison Case Description.
              Location: Cabauw, Netherlands (51.97 N, 4.93 E).
              Domain 800x800x800 m, 128^3 grid, 9-hour simulation.
              Simulation period: 00 UTC to 09 UTC, 2 July 2006.
              Surface BC: observed 0.25 m potential temperature (zTemperature=0.25).
              Geostrophic wind: time- and height-varying (optGeoWind=1).
              Large-scale advection: time- and height-varying (optAdvection=1).
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
l_x = 800
l_y = 800
l_z = 800

# Number of grid points
nx = 128
ny = 128
nz = 128

# ------------------------------------------------------------
# Time integration configuration
# ------------------------------------------------------------

# Change this if it is a restart run
istep = 1

# Time stepping and simulation time
dt = 0.1           # unit: sec
SimTime = 32400    # 9 hours (00–09 UTC 2 July 2006), unit: sec

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
z0m = 0.15
z0T = 0.15   # not used when zTemperature > 0; kept for backward compatibility

# Screen-level temperature reference height (m).
# GABLS3 surface temperature observed at 0.25 m above ground.
zTemperature = 0.25

# SensibleHeatFlux: not used when optSurfBC >= 1; set to 0 as placeholder
SensibleHeatFlux = 0.0  # K m/s

# Path to surface BC file (relative to run directory)
SurfaceBCFile = 'input/SurfaceBC.npz'

# ------------------------------------------------------------
# Moisture configuration
# ------------------------------------------------------------

# optMoisture: 0 = dry run, 1 = specific humidity Q is prognostic
optMoisture = 1

# optMoistureSurfBC: 2 = time-varying prescribed surface Q (from MoistureSurfaceBCFile)
# GABLS3: 0.25 m observed Q, same height as surface temperature
optMoistureSurfBC = 2

# zMoisture: screen-level reference height for moisture flux denominator (m).
# Same as the surface temperature observation height.
zMoisture = 0.25

# Path to moisture surface BC file (relative to run directory)
MoistureSurfaceBCFile = 'input/MoistureSurfaceBC.npz'

# MoistureFlux: not used when optMoistureSurfBC >= 2; set to 0 as placeholder
MoistureFlux = 0.0   # kg/kg m/s

# q_inversion: specific humidity lapse rate above domain top (kg/kg/m).
# Zero gradient = free-atmosphere Q matches domain top value.
q_inversion = 0.0

# ------------------------------------------------------------
# Forcing configuration
# ------------------------------------------------------------

# Geostrophic wind option:
#   0 = constant Ug2, Vg2 from Config
#   1 = time + height varying, loaded from GeoWindFile
optGeoWind = 1

# Constant geostrophic wind (m/s) — used only when optGeoWind = 0
Ug2 = -6.5
Vg2 =  4.5

# Path to geostrophic wind file (relative to run directory)
GeoWindFile = 'input/GeoWind.npz'

# Large-scale advection option:
#   0 = no mesoscale advection forcing
#   1 = time + height varying, loaded from AdvectionFile
optAdvection = 1

# Path to advection forcing file (relative to run directory)
AdvectionFile = 'input/AdvForcing.npz'

# Coriolis parameter: f = 2*Omega*sin(lat)
# Cabauw, Netherlands (~51.97 deg N)  =>  f > 0
f_coriolis = 1.149e-4   # unit: 1/s

# Potential temperature lapse rate above domain top (K/m)
inversion = 0.0029

# Buoyancy calculation: 0 = use reference T_0, 1 = use local mean THv
optBuoyancy = 1

# Reference temperature (K)
# Approximate initial near-surface potential temperature at t=0
T_0 = 290.0

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model: 0 = Static SM, 1 = LASDD-SM, 2 = LASDD-WL, 3 = LAD-SM, 4 = LAD-WL
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
z_damping  = 550.0   # unit: m  (document recommends 550-600 m)
RelaxTime  = 600.0   # unit: s

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

SampleInterval_sec   = 10.0       # collect a sample every 10 s
OutputInterval_sec   = 300.0      # output averaged stats every 5 min
Output3DInterval_sec = 3600.0     # output 3D fields every 1 hr
