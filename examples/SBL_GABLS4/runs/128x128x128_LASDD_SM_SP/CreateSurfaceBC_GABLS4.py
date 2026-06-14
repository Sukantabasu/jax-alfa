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
File: CreateSurfaceBC_GABLS4.py
=================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-06-14
:Description: Creates input/SurfaceBC.npz for the GABLS4 case.

              Reads the 36-hourly surface (skin) temperature Tg from
              GABLS4_SCM_LES_STAGE3.nc and converts it to surface
              potential temperature:
                Ths = Tg * (p0 / psurf)^0.286
              where p0 = 100000 Pa and psurf is the scalar surface
              pressure from the netCDF.

              Linearly interpolates from 37 hourly values onto every dt
              timestep (total nsteps+1 values).

              Output: input/SurfaceBC.npz
                data_series  (nsteps+1,)  float64  Ths in K
                t_series     (nsteps+1,)  float64  time axis (s)
                dt_surf      scalar       = dt
                optSurfBC    scalar       = 2
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
nsteps  = int(np.ceil(SimTime / dt))

nc_path = os.path.join(_script_dir, 'GABLS4_SCM_LES_STAGE3.nc')
ds = nc.Dataset(nc_path)
t_nc  = np.array(ds['time'][:])   # (37,) s
Tg    = np.array(ds['Tg'][:])     # (37,) K   skin temperature
psurf = float(ds['psurf'][:])     # scalar Pa
ds.close()

# Convert surface temperature to potential temperature
p0  = 100000.0  # reference pressure (Pa)
Ths = Tg * (p0 / psurf) ** 0.286

# Interpolate from hourly to dt
t_tgt = np.arange(nsteps + 1) * dt
if t_tgt[-1] > t_nc[-1] + 1e-6:
    raise ValueError(
        f"Simulation end {t_tgt[-1]:.0f} s exceeds surface BC data end {t_nc[-1]:.0f} s."
    )

_interp = interp1d(t_nc, Ths, kind='linear')
Ths_series = _interp(t_tgt)

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

out_path = os.path.join(input_dir, 'SurfaceBC.npz')
np.savez(out_path,
         data_series = Ths_series.astype(np.float64),
         t_series    = t_tgt.astype(np.float64),
         dt_surf     = np.float64(dt),
         optSurfBC   = np.int32(2))

print(f"Surface BC file written to {out_path}")
print(f"  psurf = {psurf:.0f} Pa  (p0/psurf)^0.286 = {(p0/psurf)**0.286:.5f}")
print(f"  nsteps+1 = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  Tg range:  [{Tg.min():.3f}, {Tg.max():.3f}] K")
print(f"  Ths range: [{Ths.min():.3f}, {Ths.max():.3f}] K")
print(f"  Ths(t=0)  = {Ths_series[0]:.4f} K")
print(f"  Ths(t=36h)= {Ths_series[-1]:.4f} K")
