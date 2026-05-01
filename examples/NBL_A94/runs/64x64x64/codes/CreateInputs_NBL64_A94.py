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
File: CreateInputs_NBL64_A94.py
===============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-10-22
:Description: run this file to create input files for NBL_A94 case
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(0)

# ============================================================
# Input Parameters
# ============================================================

# Domain height (m)
l_z = 4000

# Number of grid points
nx = 64
ny = 64
nz = 64

# Geostrophic wind (m/s)
Ug = 5

# Initial boundary layer height (m)
z_i = 4000

# Inversion strength (K/m)
inversion = 0/1000

# Create vertical grid
z = np.array([(k + 0.5) * l_z / (nz - 1) for k in range(nz)])

# Random perturbation amplitudes
ru = 0.03  # for velocity components
rt = 1e-3  # for temperature

# Initialize arrays
u = Ug * np.ones((nx, ny, nz))
v = np.zeros((nx, ny, nz))
w = np.zeros((nx, ny, nz))
TH = 300 * np.ones((nx, ny, nz))

# Apply perturbations for levels below 200 m
for k in range(nz):

    if z[k] <= 200:

        # Generate random perturbations
        r1 = ru * np.random.randn(nx, ny)
        r2 = rt * np.random.randn(nx, ny)

        # Apply to velocity components and temperature
        u[:, :, k] = u[:, :, k] + r1
        TH[:, :, k] = TH[:, :, k] + r2

    # Apply temperature inversion above boundary layer
    if z[k] >= z_i:
        TH[:, :, k] = TH[:, :, k] + (z[k] - z_i) * inversion

# Reshape arrays to format needed for output files
u_flat = u.reshape(-1, order='F')
v_flat = v.reshape(-1, order='F')
w_flat = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')

# Stack velocity components for velocity input file
vel_data = np.column_stack([u_flat, v_flat, w_flat])

# Save to files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'input')
os.makedirs(input_dir, exist_ok=True)
np.savetxt(os.path.join(input_dir, 'vel.ini'), vel_data)
np.savetxt(os.path.join(input_dir, 'TH.ini'), TH_flat)

print(f"Generated velocity field with shape: {vel_data.shape}")
print(f"Generated temperature field with shape: {TH_flat.shape}")
