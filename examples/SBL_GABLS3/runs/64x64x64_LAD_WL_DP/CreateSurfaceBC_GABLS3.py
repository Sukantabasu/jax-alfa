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
File: CreateSurfaceBC_GABLS3.py
=================================

:Author: Sukanta Basu
:Date: 2026-05-19
:Description: Constructs the time-varying surface boundary conditions for the
              GABLS3 case from the prescribed 0.25 m observations
              (Table 1 of GABLS3_LES_Revised.docx).

              Simulation period: 00 UTC to 09 UTC, 2 July 2006.

              Source data (Table 1): hourly values of theta_0.25 (K) and
              Q_0.25 (kg/kg) at UTC hours 0-9.

              Interpolation:
                Linear, from hourly observations onto every dt from
                t=0 to t=SimTime (total nsteps+1 values).

              Output files:
                input/SurfaceBC.npz
                  data_series  (nsteps+1,)  float64  dimensional (K)
                  dt_surf      scalar       = dt
                  optSurfBC    scalar       = 2  (prescribed surface temperature)

                input/MoistureSurfaceBC.npz
                  data_series  (nsteps+1,)  float64  dimensional (kg/kg)
                  dt_moist     scalar       = dt
                  optMoistureSurfBC  scalar = 2  (prescribed surface Q)

              Place this script in the run directory alongside Config.py.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os
from scipy.interpolate import interp1d

# ============================================================
# Read simulation parameters from Config.py
# ============================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

dt      = float(_cfg['dt'])
SimTime = float(_cfg['SimTime'])
nsteps  = int(np.ceil(SimTime / dt))

# ============================================================
# Surface BC data (Table 1)
# theta_0.25: observed potential temperature at z=0.25 m
# Time axis: UTC hours 0-9 => t = 0, 3600, ..., 32400 s
# ============================================================

t_src = np.arange(10) * 3600.0     # [0, 3600, ..., 32400] s

theta_025 = np.array([
    291.28,   # t=0h  (00 UTC)
    290.34,   # t=1h
    289.45,   # t=2h
    288.62,   # t=3h
    288.43,   # t=4h
    289.95,   # t=5h
    292.38,   # t=6h
    294.16,   # t=7h
    296.55,   # t=8h
    298.45,   # t=9h  (09 UTC)
], dtype=float)

# Table 1: Q at z=0.25 m (kg/kg)
Q_025 = np.array([
    0.0100,   # t=0h  (00 UTC)
    0.0099,   # t=1h
    0.0100,   # t=2h
    0.0099,   # t=3h
    0.0099,   # t=4h
    0.0104,   # t=5h
    0.0109,   # t=6h
    0.0113,   # t=7h
    0.0121,   # t=8h
    0.0129,   # t=9h  (09 UTC)
], dtype=float)

# ============================================================
# Validate coverage
# ============================================================

t_tgt = np.arange(nsteps + 1) * dt

if t_tgt[-1] > t_src[-1] + 1e-6:
    raise ValueError(
        f"Simulation end time {t_tgt[-1]:.0f} s ({t_tgt[-1]/3600:.2f} h) "
        f"exceeds last surface BC observation at {t_src[-1]:.0f} s "
        f"({t_src[-1]/3600:.2f} h)."
    )

# ============================================================
# Interpolate to simulation time axis
# ============================================================

_interp_TH   = interp1d(t_src, theta_025, kind='linear')
_interp_Q    = interp1d(t_src, Q_025,    kind='linear')
theta_series = _interp_TH(t_tgt)   # (nsteps+1,) K
Q_series     = _interp_Q(t_tgt)    # (nsteps+1,) kg/kg

# ============================================================
# Save
# ============================================================

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

out_path = os.path.join(input_dir, 'SurfaceBC.npz')
np.savez(out_path,
         data_series = theta_series.astype(np.float64),
         dt_surf     = np.float64(dt),
         optSurfBC   = np.int32(2))

out_path_q = os.path.join(input_dir, 'MoistureSurfaceBC.npz')
np.savez(out_path_q,
         data_series      = Q_series.astype(np.float64),
         dt_moist         = np.float64(dt),
         optMoistureSurfBC = np.int32(2))

print(f"SurfaceBC file written to {out_path}")
print(f"  nsteps+1 = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  Source: {len(t_src)} hourly observations (00-09 UTC)")
print(f"  theta_0.25 range: [{theta_series.min():.3f}, {theta_series.max():.3f}] K")
print(f"  theta at t=0 h : {theta_series[0]:.4f} K")
print(f"  theta at t=4 h : {theta_series[int(4*3600/dt)]:.4f} K  (minimum)")
print(f"  theta at t=9 h : {theta_series[-1]:.4f} K")
print(f"MoistureSurfaceBC file written to {out_path_q}")
print(f"  Q_0.25 range: [{Q_series.min():.5f}, {Q_series.max():.5f}] kg/kg")
print(f"  Q at t=0 h   : {Q_series[0]:.5f} kg/kg")
print(f"  Q at t=9 h   : {Q_series[-1]:.5f} kg/kg")
