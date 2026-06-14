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
File: CreateInputs_SBL_GABLS4.py
=================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-06-14
:Description: Creates vel.npy and TH.npy for the SBL_GABLS4 case.
              Reads initial u, v, theta profiles from GABLS4_SCM_LES_STAGE3.nc
              (Stage 3 initialization data), interpolates onto model
              half-levels, and adds random perturbations below z = 100 m.

              Perturbation amplitudes (from GABLS4_Initialization.m):
                u, w : 0.1 (m/s)   for z <= 100 m
                TH   : 0.1 (K)     for z <= 100 m

              Output (written to input/ sub-directory):
                vel.npy   dimensional (u, v, w) in m/s, Fortran order
                TH.npy    dimensional potential temperature in K, Fortran order
"""

import numpy as np
import os
from scipy.interpolate import interp1d

try:
    import netCDF4 as nc
except ImportError:
    raise ImportError("netCDF4 package required: pip install netCDF4")

np.random.seed(42)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

nx  = int(_cfg['nx'])
ny  = int(_cfg['ny'])
nz  = int(_cfg['nz'])
l_z = float(_cfg['l_z'])

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])

nc_path = os.path.join(_script_dir, 'GABLS4_SCM_LES_STAGE3.nc')
ds = nc.Dataset(nc_path)
height = np.array(ds['height'][:])
theta  = np.array(ds['theta'][:])
u_nc   = np.array(ds['u'][:])
v_nc   = np.array(ds['v'][:])
ds.close()

# Sort by height ascending (netCDF is top-to-bottom)
idx    = np.argsort(height)
h_s    = height[idx]
th_s   = theta[idx]
u_s    = u_nc[idx]
v_s    = v_nc[idx]

# Interpolate profiles onto model half-levels
_fu  = interp1d(h_s, u_s,  kind='linear', fill_value='extrapolate')
_fv  = interp1d(h_s, v_s,  kind='linear', fill_value='extrapolate')
_fth = interp1d(h_s, th_s, kind='linear', fill_value='extrapolate')

U1D  = _fu(z)
V1D  = _fv(z)
TH1D = _fth(z)

u  = np.zeros((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = np.zeros((nx, ny, nz))

for k in range(nz):
    u[:, :, k]  = U1D[k]
    v[:, :, k]  = V1D[k]
    TH[:, :, k] = TH1D[k]

# Random perturbations below z = 100 m
for k in range(nz):
    if z[k] <= 100.0:
        R1 = 0.1 * np.random.randn(nx, ny)
        u[:, :, k]  += R1
        w[:, :, k]  += R1
        TH[:, :, k] += 0.1 * np.random.randn(nx, ny)

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')
vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)
np.save(os.path.join(input_dir, 'vel.npy'), vel_data)
np.save(os.path.join(input_dir, 'TH.npy'),  TH_flat)

print(f"GABLS4 initial conditions written to {input_dir}")
print(f"  Grid: nx={nx}, ny={ny}, nz={nz}, dz={dz:.4f} m")
print(f"  vel.npy shape: {vel_data.shape}")
print(f"  TH.npy  shape: {TH_flat.shape}")
print(f"  U range: [{U1D.min():.3f}, {U1D.max():.3f}] m/s")
print(f"  V range: [{V1D.min():.3f}, {V1D.max():.3f}] m/s")
print(f"  TH range: [{TH1D.min():.3f}, {TH1D.max():.3f}] K")
