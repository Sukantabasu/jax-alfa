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
File: CreateGeoWind_Wangara.py
================================

:Author: Sukanta Basu
:Date: 2026-05-18
:Description: Constructs the time- and height-varying geostrophic wind for the
              Wangara diurnal cycle case (Basu et al. 2008, BLM) directly from
              the embedded observational data — no external Ugt/Vgt files needed.

              Simulation period: full diurnal cycle from 0900 LST day 33
              (16 August 1967) to 0900 LST day 34 (17 August 1967).

              Method (translated from Wangara_Initialization.m):
                1. Surface geostrophic wind Ug0, Vg0: 9 values at 3-hourly
                   intervals (0, 3, ..., 24 h).
                2. Thermal wind increments dUg at z=1000 m and z=2000 m:
                   defined at 3 12-hourly times (0, 12, 24 h), linearly
                   interpolated to 3-hourly.
                3. For each 3-hourly snapshot, fit a quadratic vertical profile
                     Ug(z) = a·z² + b·z + Ug0
                   using Ug at z=0, 1000 m, and 2000 m as constraints.
                4. Evaluate the profile at the model half-levels from Config.py.
                5. Linearly interpolate from 3-hourly to every dt (nsteps+1 pts).

              Note: the original MATLAB (line 92) computes dVg2F using dVg1
              instead of dVg2 — this appears to be a typo but is replicated
              here for consistency with the published Wangara results.

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
# Read simulation parameters from Config.py (same directory)
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
# Source time axes
# ============================================================

# 3-hourly: 9 points spanning 0–24 h
t_3h = np.arange(9) * 3.0 * 3600.0        # [0, 10800, ..., 86400] s

# 12-hourly: 3 points (0, 12, 24 h) — used for thermal wind interpolation
t_12h = np.arange(3) * 12.0 * 3600.0      # [0, 43200, 86400] s

# ============================================================
# Thermal wind increments (defined at 12-hourly, interpolated to 3-hourly)
# Heights: z1=1000 m, z2=2000 m above the surface geostrophic wind level
# ============================================================

dUg1_12h = np.array([2.98,  2.82,  1.87])   # increment at z1=1000 m
dUg2_12h = np.array([1.49,  1.32,  1.04])   # increment at z2=2000 m

dVg1_12h = np.array([-0.04, -0.67,  0.59])  # increment at z1=1000 m
# Note: original MATLAB (line 92) mistakenly uses dVg1 values for dVg2F.
# Replicated here for consistency with published Wangara results.
dVg2_12h = dVg1_12h.copy()                  # intentional copy of dVg1

# Interpolate to 3-hourly
dUg1 = np.interp(t_3h, t_12h, dUg1_12h)    # (9,)
dUg2 = np.interp(t_3h, t_12h, dUg2_12h)    # (9,)
dVg1 = np.interp(t_3h, t_12h, dVg1_12h)    # (9,)
dVg2 = np.interp(t_3h, t_12h, dVg2_12h)    # (9,)

# ============================================================
# Surface geostrophic wind (3-hourly, 9 values: 0–24 h)
# ============================================================

Ug0 = np.array([-5.34, -5.56, -6.20, -6.42, -5.99, -6.93, -8.02, -7.32, -7.60])
Vg0 = np.array([-0.43,  1.27,  0.98, -0.72, -1.93, -2.86, -3.43, -4.97, -4.72])

# Geostrophic wind at z1=1000 m and z2=2000 m (3-hourly)
Ug1 = Ug0 + dUg1
Ug2 = Ug0 + dUg1 + dUg2

Vg1 = Vg0 + dVg1
Vg2 = Vg0 + dVg1 + dVg2

# ============================================================
# Model vertical half-levels (from Config.py)
# ============================================================

dz   = l_z / (nz - 1)
z    = np.array([(k + 0.5) * dz for k in range(nz)])   # (nz,) m

# ============================================================
# Quadratic vertical profile per 3-hourly snapshot
# Ug(z) = a·z² + b·z + Ug0[j]
# Constraints at z1=1000 m and z2=2000 m
# ============================================================

z1_ref = 1000.0   # m
z2_ref = 2000.0   # m
A = np.array([[z1_ref**2, z1_ref],
              [z2_ref**2, z2_ref]])

Ug_3h = np.zeros((9, nz))
Vg_3h = np.zeros((9, nz))

for j in range(9):
    RU = np.array([Ug1[j] - Ug0[j], Ug2[j] - Ug0[j]])
    RV = np.array([Vg1[j] - Vg0[j], Vg2[j] - Vg0[j]])

    LU = np.linalg.solve(A, RU)
    LV = np.linalg.solve(A, RV)

    Ug_3h[j, :] = LU[0] * z**2 + LU[1] * z + Ug0[j]
    Vg_3h[j, :] = LV[0] * z**2 + LV[1] * z + Vg0[j]

# ============================================================
# Interpolate from 3-hourly to every dt  ->  (nsteps+1, nz)
# ============================================================

t_tgt = np.arange(nsteps + 1) * dt        # (nsteps+1,) s

if t_tgt[-1] > t_3h[-1] + 1e-6:
    raise ValueError(
        f"Simulation end time {t_tgt[-1]:.0f} s ({t_tgt[-1]/3600:.1f} h) "
        f"exceeds the 3-hourly source data end {t_3h[-1]:.0f} s (24.0 h)."
    )

Ug_series = np.zeros((nsteps + 1, nz))
Vg_series = np.zeros((nsteps + 1, nz))

for k in range(nz):
    Ug_series[:, k] = np.interp(t_tgt, t_3h, Ug_3h[:, k])
    Vg_series[:, k] = np.interp(t_tgt, t_3h, Vg_3h[:, k])

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
print(f"  nz = {nz},  dz = {dz:.3f} m,  l_z = {l_z} m")
print(f"  Ug range: [{Ug_series.min():.3f}, {Ug_series.max():.3f}] m/s")
print(f"  Vg range: [{Vg_series.min():.3f}, {Vg_series.max():.3f}] m/s")
print(f"  Ug at t=0h,  z[0]={z[0]:.2f}m : {Ug_series[0,  0]:.4f} m/s  "
      f"(source: {Ug_3h[0, 0]:.4f} m/s)")
print(f"  Ug at t=12h, z[0]={z[0]:.2f}m : {Ug_series[int(12*3600/dt), 0]:.4f} m/s  "
      f"(source: {Ug_3h[4, 0]:.4f} m/s)")
