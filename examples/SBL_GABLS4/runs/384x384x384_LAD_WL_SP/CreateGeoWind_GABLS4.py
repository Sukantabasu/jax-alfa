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
File: CreateGeoWind_GABLS4.py
===============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-06-14
:Description: Constructs the time- and height-varying geostrophic wind for
              the GABLS4 case from GABLS4_SCM_LES_STAGE3.nc.

              The netCDF contains hourly Ug/Vg profiles (37 time steps,
              t = 0 to 129600 s) at 90 pressure levels.  This script:
                1. Sorts levels by height ascending.
                2. Interpolates spatially onto model half-levels (nz).
                3. Interpolates temporally from hourly to every dt.

              Output: input/GeoWind.npz
                Ug_series  (nsteps+1, nz)  float64  dimensional (m/s)
                Vg_series  (nsteps+1, nz)  float64  dimensional (m/s)
                dt_geo     scalar          = dt
                optGeoWind scalar          = 1
"""

import numpy as np
import os
from scipy.interpolate import interp1d

try:
    import netCDF4 as nc
except ImportError:
    raise ImportError("netCDF4 package required: pip install netCDF4")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

dt      = float(_cfg['dt'])
SimTime = float(_cfg['SimTime'])
nz      = int(_cfg['nz'])
l_z     = float(_cfg['l_z'])

nsteps = int(np.ceil(SimTime / dt))

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])

nc_path = os.path.join(_script_dir, 'GABLS4_SCM_LES_STAGE3.nc')
ds = nc.Dataset(nc_path)
t_nc   = np.array(ds['time'][:])    # (37,) s
height = np.array(ds['height'][:])  # (90,) m
Ug_nc  = np.array(ds['Ug'][:])     # (37, 90)
Vg_nc  = np.array(ds['Vg'][:])     # (37, 90)
ds.close()

# Sort heights ascending
idx_h = np.argsort(height)
h_s   = height[idx_h]
Ug_nc = Ug_nc[:, idx_h]   # (37, 90)
Vg_nc = Vg_nc[:, idx_h]

# Step 1: Interpolate spatially onto model half-levels for each of the 37 time points
Ug_model = np.zeros((len(t_nc), nz))
Vg_model = np.zeros((len(t_nc), nz))
for it in range(len(t_nc)):
    _fu = interp1d(h_s, Ug_nc[it], kind='linear', fill_value='extrapolate')
    _fv = interp1d(h_s, Vg_nc[it], kind='linear', fill_value='extrapolate')
    Ug_model[it] = _fu(z)
    Vg_model[it] = _fv(z)

# Step 2: Interpolate temporally from hourly to every dt
t_tgt = np.arange(nsteps + 1) * dt
if t_tgt[-1] > t_nc[-1] + 1e-6:
    raise ValueError(
        f"Simulation end {t_tgt[-1]:.0f} s exceeds GeoWind data end {t_nc[-1]:.0f} s."
    )

Ug_series = np.zeros((nsteps + 1, nz))
Vg_series = np.zeros((nsteps + 1, nz))
for k in range(nz):
    _itu = interp1d(t_nc, Ug_model[:, k], kind='linear')
    _itv = interp1d(t_nc, Vg_model[:, k], kind='linear')
    Ug_series[:, k] = _itu(t_tgt)
    Vg_series[:, k] = _itv(t_tgt)

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

out_path = os.path.join(input_dir, 'GeoWind.npz')
np.savez(out_path,
         Ug_series  = Ug_series.astype(np.float64),
         Vg_series  = Vg_series.astype(np.float64),
         dt_geo     = np.float64(dt),
         optGeoWind = np.int32(1))

print(f"GeoWind file written to {out_path}")
print(f"  nsteps+1 = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  nz = {nz},  dz = {dz:.4f} m")
print(f"  Ug range: [{Ug_series.min():.3f}, {Ug_series.max():.3f}] m/s")
print(f"  Vg range: [{Vg_series.min():.3f}, {Vg_series.max():.3f}] m/s")
print(f"  Ug at t=0, z[0]={z[0]:.2f} m: {Ug_series[0, 0]:.4f} m/s")
print(f"  Vg at t=0, z[0]={z[0]:.2f} m: {Vg_series[0, 0]:.4f} m/s")
