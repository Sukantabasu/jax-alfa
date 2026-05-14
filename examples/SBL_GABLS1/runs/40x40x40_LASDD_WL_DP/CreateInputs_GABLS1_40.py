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
File: CreateInputs_GABLS1_40.py
================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-08
:Description: Creates vel.ini and TH.ini for the GABLS1 stable boundary
              layer case on a 40x40x40 grid.

              Initial conditions (Beare et al. 2006):
                u = Ug = 8 m/s, v = 0 m/s (plus small perturbations)
                TH = 265 K for z <= 100 m
                TH = 265 + 0.01*(z-100) K for z > 100 m
              Small random perturbations added below 50 m.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os

np.random.seed(42)

# ============================================================
# Parameters (must match Config.py)
# ============================================================

l_z = 400   # domain height (m)
nx  = 40
ny  = 40
nz  = 40

Ug           = 8.0    # geostrophic wind (m/s)
T_sfc_init   = 265.0  # initial surface temperature (K)
inversion    = 0.01   # K/m, above 100 m

# Random perturbation amplitudes
rw = 0.01  # for velocity components
rt = 0.01  # for temperature

# ============================================================
# Vertical grid (half levels)
# ============================================================

dz_dim = l_z / (nz - 1)   # ~10.26 m
z = np.array([(k + 0.5) * dz_dim for k in range(nz)])

# ============================================================
# Initialise fields
# ============================================================

u  = Ug * np.ones((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = np.zeros((nx, ny, nz))

for k in range(nz):

    # Temperature profile
    if z[k] <= 100.0:
        TH[:, :, k] = T_sfc_init
    else:
        TH[:, :, k] = T_sfc_init + inversion * (z[k] - 100.0)

    # Small perturbations below 50 m to trigger turbulence
    if z[k] <= 50.0:
        w[:, :, k]  += rw * np.random.randn(nx, ny)
        TH[:, :, k] += rt  * np.random.randn(nx, ny)

# ============================================================
# Flatten and save (Fortran order to match LES reader)
# ============================================================

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')

vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(input_dir, exist_ok=True)

np.savetxt(os.path.join(input_dir, 'vel.ini'), vel_data)
np.savetxt(os.path.join(input_dir, 'TH.ini'),  TH_flat)

print(f"GABLS1 initial conditions written to {input_dir}")
print(f"  vel.ini  shape: {vel_data.shape}  (u, v, w columns)")
print(f"  TH.ini   shape: {TH_flat.shape}")
print(f"  Vertical grid: dz = {dz_dim:.3f} m, "
      f"z[0] = {z[0]:.3f} m, z[-1] = {z[-1]:.3f} m")
print(f"  TH range: {TH.min():.3f} – {TH.max():.3f} K")
