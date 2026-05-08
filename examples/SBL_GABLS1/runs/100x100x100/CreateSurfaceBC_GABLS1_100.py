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
File: CreateSurfaceBC_GABLS1_100.py
=====================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-08
:Description: Creates input/SurfaceBC.npz for the GABLS1 case (100^3 grid).

              Supports two modes (set optSurfTemp below):
                1 — Analytical linear cooling (GABLS1 standard)
                2 — Interpolation from observed data at sparse hours

              Output file: input/SurfaceBC.npz
                data_series : (nsteps+1,) float64, surface temperature (K)
                t_series    : (nsteps+1,) float64, time axis (s)
                dt_surf     : scalar, = dt (for validation at load time)
                optSurfBC   : scalar, = 2 (prescribed surface temperature)
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os
from scipy.interpolate import CubicSpline

# ============================================================
# Read dt and SimTime from Config.py (same directory)
# ============================================================

_cfg = {}
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Config.py')
with open(_cfg_path) as _f:
    exec(_f.read(), _cfg)

dt      = float(_cfg['dt'])
SimTime = float(_cfg['SimTime'])

# ============================================================
# Surface BC mode
# ============================================================

# optSurfTemp = 1 : Mode A — analytical linear cooling
# optSurfTemp = 2 : Mode B — interpolation from observed sparse data
optSurfTemp = 1

# ============================================================
# Mode A: linear cooling (GABLS1 standard)
# ============================================================

T_sfc_initial = 265.0  # K, surface temperature at t = 0
cooling_rate  = 0.25   # K/hr (positive = cooling)
#   T_s(t) = T_sfc_initial - cooling_rate * (t / 3600)
#   At t = 9 h: T_s = 265 - 0.25*9 = 262.75 K

# ============================================================
# Mode B: interpolation from observed data at sparse hours
# ============================================================

# Uncomment and edit when optSurfTemp = 2
# t_obs     = np.array([0, 1, 3, 6, 9]) * 3600.0  # seconds
# T_sfc_obs = np.array([265.0, 264.75, 264.25, 263.50, 262.75])  # K
# interp_method = 'linear'   # 'linear' or 'cubic'

# ============================================================
# Build time axis at dt resolution
# ============================================================

nsteps   = int(np.ceil(SimTime / dt))
t_series = np.arange(0, nsteps + 1) * dt   # shape (nsteps+1,), seconds

# ============================================================
# Compute surface temperature series
# ============================================================

if optSurfTemp == 1:

    T_sfc = T_sfc_initial - cooling_rate * (t_series / 3600.0)

elif optSurfTemp == 2:

    if interp_method == 'linear':
        T_sfc = np.interp(t_series, t_obs, T_sfc_obs)
    else:
        cs    = CubicSpline(t_obs, T_sfc_obs, bc_type='not-a-knot')
        T_sfc = cs(t_series)

else:
    raise ValueError(f"Unknown optSurfTemp={optSurfTemp}. Use 1 or 2.")

# ============================================================
# Save
# ============================================================

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(input_dir, exist_ok=True)

out_path = os.path.join(input_dir, 'SurfaceBC.npz')
np.savez(out_path,
         data_series = T_sfc.astype(np.float64),
         t_series    = t_series.astype(np.float64),
         dt_surf     = np.float64(dt),
         optSurfBC   = np.int32(2))

print(f"Surface BC file written to {out_path}")
print(f"  Mode: {'linear cooling' if optSurfTemp == 1 else 'observed interpolation'}")
print(f"  nsteps + 1    = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  T_s(t=0)      = {T_sfc[0]:.4f} K")
print(f"  T_s(t=SimTime)= {T_sfc[-1]:.4f} K")
print(f"  T_s range     : [{T_sfc.min():.4f}, {T_sfc.max():.4f}] K")
