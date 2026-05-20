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
File: CreateSurfaceBC_Wangara.py
==================================

:Author: Sukanta Basu
:Date: 2026-05-18
:Description: Reads the Wangara Day-33 screen temperature observations
              (Wangara_Sfc3309.txt), converts them to screen-level potential
              temperature at 1.2 m, interpolates onto the simulation time axis,
              and saves input/SurfaceBC.npz for optSurfBC=2.

              Simulation period: full diurnal cycle from 0900 LST day 33
              (16 August 1967) to 0900 LST day 34 (17 August 1967).

              Source data format (Wangara_Sfc3309.txt):
                Column 1: absolute time (LST hours: 9, 10, ..., 33)
                Column 2: pressure (mb)
                Column 3: screen-level air temperature (°C) at z_T = 1.2 m

              Screen-level potential temperature (Mellor-Yamada DALR):
                theta_screen = (T_C + 273.16) + zTemperature * 10 / 1000
                where zTemperature = 1.2 m (set in Config.py)

              Interpolation:
                Linear, from the observed time resolution onto every dt
                from t=0 to t=SimTime (total nsteps+1 values).

              The simulation end time must not exceed the last observation.

              Output: input/SurfaceBC.npz
                data_series  (nsteps+1,)  float64  dimensional (K)
                dt_surf      scalar       = dt (for validation at load time)
                optSurfBC    scalar       = 2  (prescribed surface temperature)

              Place this script in the run directory alongside Config.py.
              Set SFC_DATA_FILE below if Wangara_Sfc3309.txt lives elsewhere.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os
from scipy.interpolate import interp1d

# ============================================================
# Path to surface temperature data file
# (adjust if Wangara_Sfc3309.txt is not in the same directory)
# ============================================================

_script_dir   = os.path.dirname(os.path.abspath(__file__))
SFC_DATA_FILE = os.path.join(_script_dir, 'Wangara_Sfc3309.txt')

# ============================================================
# Read simulation parameters from Config.py (same directory)
# ============================================================

_cfg = {}
_cfg_path = os.path.join(_script_dir, 'Config.py')
with open(_cfg_path) as _f:
    exec(_f.read(), _cfg)

dt           = float(_cfg['dt'])            # timestep (s)
SimTime      = float(_cfg['SimTime'])       # total simulation time (s)
zTemperature = float(_cfg.get('zTemperature', 1.2))  # screen height (m)

nsteps = int(np.ceil(SimTime / dt))

# ============================================================
# Load surface data
# ============================================================

if not os.path.isfile(SFC_DATA_FILE):
    raise FileNotFoundError(
        f"Surface data file not found: {SFC_DATA_FILE}\n"
        "Provide Wangara_Sfc3309.txt (columns: time_h, P_mb, T_screen_C)."
    )

D = np.loadtxt(SFC_DATA_FILE)

T_C = D[:, 2]                        # screen temperature (°C)

# Column 0 is absolute time (EST hours: 9, 10, ..., 33).
# Convert to seconds from simulation start by subtracting the first entry.
t_src = (D[:, 0] - D[0, 0]) * 3600.0   # (N_obs,) seconds from t=0

# Screen-level potential temperature — Mellor-Yamada DALR correction
# (matches Wangara_Initialization.m: Tscr = (D(k,3)+273.16)+1.2*10/1000)
theta_src = (T_C + 273.16) + zTemperature * 10.0 / 1000.0   # (N_obs,) K

# ============================================================
# Validate coverage
# ============================================================

t_tgt = np.arange(nsteps + 1) * dt        # (nsteps+1,) seconds

if t_tgt[-1] > t_src[-1] + 1e-6:
    raise ValueError(
        f"Simulation end time {t_tgt[-1]:.0f} s ({t_tgt[-1]/3600:.2f} h) "
        f"exceeds last observation at {t_src[-1]:.0f} s ({t_src[-1]/3600:.2f} h). "
        "Extend Wangara_Sfc3309.txt coverage."
    )

if t_tgt[0] < t_src[0] - 1e-6:
    raise ValueError(
        f"Simulation start time {t_tgt[0]:.0f} s precedes first observation "
        f"at {t_src[0]:.0f} s. Prepend earlier data or adjust Config."
    )

# ============================================================
# Interpolate to simulation time axis
# ============================================================

_interp = interp1d(t_src, theta_src, kind='linear', fill_value='extrapolate')
theta_series = _interp(t_tgt)              # (nsteps+1,) K

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

print(f"SurfaceBC file written to {out_path}")
print(f"  nsteps+1 = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  zTemperature = {zTemperature} m  (screen height for MOST)")
print(f"  Source: {len(t_src)} observations, "
      f"t=[{t_src[0]/3600:.2f}, {t_src[-1]/3600:.2f}] h")
print(f"  theta_screen range: [{theta_series.min():.3f}, {theta_series.max():.3f}] K")
print(f"  theta at t=0 h : {theta_series[0]:.4f} K")
print(f"  theta at t=12 h: {theta_series[int(12*3600/dt)]:.4f} K")
