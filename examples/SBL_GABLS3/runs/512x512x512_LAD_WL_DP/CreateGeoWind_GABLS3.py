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
File: CreateGeoWind_GABLS3.py
===============================

:Author: Sukanta Basu
:Date: 2026-05-19
:Description: Constructs the time- and height-varying geostrophic wind for
              the GABLS3 case (GABLS3_LES_Revised.docx, Table 4).

              Simulation period: 00 UTC to 09 UTC, 2 July 2006.

              Method (from case description):
                1. Linearly interpolate the surface geostrophic wind
                   (Table 4, 4 time points) to every dt.
                2. At each time step, build a linear vertical profile
                   from the surface value to fixed values at z=2000 m:
                     Ugeo(z=2000 m) = -2.0 m/s
                     Vgeo(z=2000 m) =  2.0 m/s
                3. Evaluate at model half-levels (0 to l_z).

              Table 4 time points (all times relative to 00 UTC 2 July 2006):
                20060701 23:00  =>  t = -3600 s   Ugeo=-6.5  Vgeo=4.5
                20060702 03:00  =>  t = +10800 s  Ugeo=-5.0  Vgeo=4.5
                20060702 06:00  =>  t = +21600 s  Ugeo=-5.0  Vgeo=4.5
                20060702 12:00  =>  t = +43200 s  Ugeo=-6.5  Vgeo=2.5

              Output: input/GeoWind.npz
                Ug_series  (nsteps+1, nz)  float64  dimensional (m/s)
                Vg_series  (nsteps+1, nz)  float64  dimensional (m/s)
                dt_geo     scalar          = dt (for validation at load time)
                optGeoWind scalar          = 1 (time + height varying)

              Place this script in the run directory alongside Config.py.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os

# ============================================================
# Read simulation parameters from Config.py
# ============================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

dt      = float(_cfg['dt'])
SimTime = float(_cfg['SimTime'])
nz      = int(_cfg['nz'])
l_z     = float(_cfg['l_z'])

nsteps = int(np.ceil(SimTime / dt))

# ============================================================
# Model vertical half-levels
# ============================================================

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])   # (nz,) m

# ============================================================
# Surface geostrophic wind (Table 4)
# Times in seconds relative to 00 UTC 2 July 2006
# ============================================================

t_src   = np.array([-3600.0, 10800.0, 21600.0, 43200.0])  # s
Ug_sfc  = np.array([ -6.5,   -5.0,    -5.0,    -6.5  ])   # m/s
Vg_sfc  = np.array([  4.5,    4.5,     4.5,     2.5   ])   # m/s

# Fixed geostrophic wind at z=2000 m (case description)
Ug_top = -2.0   # m/s
Vg_top =  2.0   # m/s
z_top  = 2000.0  # m

# ============================================================
# Validate coverage
# ============================================================

t_tgt = np.arange(nsteps + 1) * dt     # (nsteps+1,) s

if t_tgt[-1] > t_src[-1] + 1e-6:
    raise ValueError(
        f"Simulation end time {t_tgt[-1]:.0f} s ({t_tgt[-1]/3600:.2f} h) "
        f"exceeds geostrophic wind data end {t_src[-1]:.0f} s "
        f"({t_src[-1]/3600:.2f} h)."
    )

# ============================================================
# Step 1: Interpolate surface geostrophic wind to every dt
# ============================================================

Ug_sfc_series = np.interp(t_tgt, t_src, Ug_sfc)   # (nsteps+1,)
Vg_sfc_series = np.interp(t_tgt, t_src, Vg_sfc)   # (nsteps+1,)

# ============================================================
# Step 2: Build linear vertical profile at each time step
# Ugeo(z, t) = Ug_sfc(t) + (Ug_top - Ug_sfc(t)) * z / z_top
# ============================================================

# weight shape: (nz,) — vertical interpolation factor
w_z = z / z_top        # 0 at z=0, 1 at z=z_top=2000 m

# Broadcasting: Ug_sfc_series is (nsteps+1,), w_z is (nz,)
# Ug_series[t, k] = Ug_sfc_series[t] + (Ug_top - Ug_sfc_series[t]) * w_z[k]
Ug_series = (Ug_sfc_series[:, np.newaxis]
             + (Ug_top - Ug_sfc_series[:, np.newaxis]) * w_z[np.newaxis, :])

Vg_series = (Vg_sfc_series[:, np.newaxis]
             + (Vg_top - Vg_sfc_series[:, np.newaxis]) * w_z[np.newaxis, :])

# ============================================================
# Save
# ============================================================

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
print(f"  nz = {nz},  dz = {dz:.4f} m,  l_z = {l_z} m")
print(f"  Ug range: [{Ug_series.min():.3f}, {Ug_series.max():.3f}] m/s")
print(f"  Vg range: [{Vg_series.min():.3f}, {Vg_series.max():.3f}] m/s")
print(f"  Ug surface at t=0h : {Ug_sfc_series[0]:.4f} m/s")
print(f"  Ug at t=0h,  z[0]  : {Ug_series[0,  0]:.4f} m/s")
print(f"  Ug at t=3h,  z[0]  : {Ug_series[int(3*3600/dt), 0]:.4f} m/s")
