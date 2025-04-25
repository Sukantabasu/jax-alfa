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
:Date: 2025-4-3
:Description: namelist file for JAXLES runs
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

# ------------------------------------------------------------
# Domain configuration
# ------------------------------------------------------------

# Domain size (m)
l_x = 6400
l_y = 6400
l_z = 5000

# Number of grid points
nx = 96
ny = 96
nz = 96

# ------------------------------------------------------------
# Time integration configuration
# ------------------------------------------------------------

# Change this if it is a restart run
istep = 1

# Time stepping and simulation time
dt = 2  # unit: sec
SimTime = 3.5*3600  # unit: sec

# Galilean transformation
Ugal = 0  # unit: m/s

# ------------------------------------------------------------
# Surface configuration
# ------------------------------------------------------------

# 0: homogeneous, prescribed-flux
# 1: heterogeneous, prescribed-flux
optSurfFlux = 1

# Roughness lengths
z0m = 0.16  # unit: m
z0T = 0.16  # unit: m

SensibleHeatFlux = 0.06 # K m/s


# ------------------------------------------------------------
# Forcing configuration
# ------------------------------------------------------------

# Geostrophic wind speeds
Ug2 = 0  # unit: m/s
Vg2 = 0  # unit: m/s

# Coriolis term
f_coriolis = 1e-4  # unit: 1/s

# Inversion strength
inversion = 3/1000  # unit: K/m

# Buoyancy calculation
optBuoyancy = 2  #1: use reference T_0, 2: use local THv

# Reference temperature
T_0 = 300  # unit: K

# ------------------------------------------------------------
# Subgrid-scale configuration
# ------------------------------------------------------------

# SGS model to use
# 0: Static Smagorinsky
# 1: Locally-averaged Dynamic (LAD) - Smagorinsky
# 2: Locally-averaged Scale-dependent Dynamic (LASDD) - Smagorisky
optSgs = 2

# How often dynamic SGS coefficients are computed?
dynamicSGS_call_time = 1

# Filter to grid ratio (recommended: FGR = 2)
# Dealiasing is not activated for FGR >= 2
FGR = 2  # FGR = 1 implies implicit filtering

# Initialize Cs2 and Cs2PrRatio for static SGS models
Cs2 = 0.1 ** 2
Cs2PrRatio = Cs2 / 1.0

# ------------------------------------------------------------
# Damping layer configuration
# ------------------------------------------------------------

# Rayleigh damping to activate?
optDamping = 1  # 1: yes, 0: no

# Starting height of damping layer
z_damping = 3300  # unit: m

# ------------------------------------------------------------
# Statistics computation
# ------------------------------------------------------------

# Collect samples every SampleInterval_sec (sec)
SampleInterval_sec = 10.0 # ideally, should be divisible by dt
# Output averages every OutputInterval_sec (sec)
OutputInterval_sec = 60.0 # ideally, should be divisible by dt
# Output 3D fields every Output3DInterval_sec (sec)
Output3DInterval_sec = 600.0 # ideally, should be divisible by dt
