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
File: CreateAdvForcing_GABLS3.py
==================================

:Author: Sukanta Basu
:Date: 2026-05-19
:Description: Constructs the time- and height-varying large-scale advection
              forcing for the GABLS3 case (GABLS3_LES_Revised.docx, Tables 5-7).

              Simulation period: 00 UTC to 09 UTC, 2 July 2006.

              Sign convention (from case description):
                dU/dt|_adv = Uadv   [m/s^2]
                dV/dt|_adv = Vadv   [m/s^2]
                dtheta/dt|_adv = THadv  [K/s]

              Vertical profile (from case description):
                Tendencies are constant in z = 200-800 m.
                Below 200 m, linearly interpolated to zero at z = 0 m.
                  F(z) = F_col * min(z/200, 1)

              Step changes in the time series are represented by duplicate
              time entries: the "before" value at t_jump - 1 ms and the
              "after" value at t_jump itself, so np.interp produces an
              effectively instantaneous transition that takes effect at the
              exact table time (not one step later).

              Table 5 — Uadv, Vadv [m/s^2]  (times relative to 00 UTC 2 July 2006):
                20060701 23:00  t=-3600 s   Uadv=5.0e-4  Vadv=0.0
                20060702 03:00  t=10800 s   Uadv=5.0e-4  [before step]
                20060702 03:00  t=10800 s   Uadv=0.0     [after step]
                20060702 12:00  t=43200 s   Uadv=0.0

              Table 6 — THadv [K/s]:
                20060701 12:00  t=-43200 s  THadv=-2.5e-5  (pre-simulation)
                20060702 01:00  t=3600 s    THadv=-2.5e-5  [before step]
                20060702 01:00  t=3600 s    THadv=+7.5e-5  [after step]
                20060702 06:00  t=21600 s   THadv=+7.5e-5  [before step]
                20060702 06:00  t=21600 s   THadv=0.0      [after step]
                20060702 12:00  t=43200 s   THadv=0.0

              Output: input/AdvForcing.npz
                Uadv_series   (nsteps+1, nz)  float64  [m/s^2]
                Vadv_series   (nsteps+1, nz)  float64  [m/s^2]
                THadv_series  (nsteps+1, nz)  float64  [K/s]
                Qadv_series   (nsteps+1, nz)  float64  [kg/kg/s]
                dt_adv        scalar          = dt
                optAdvection  scalar          = 1

              Table 7 — Qadv [kg/kg/s]  (times relative to 00 UTC 2 July 2006):
                20060702 00:00  t=0 s       Qadv=0.0
                20060702 02:00  t=7200 s    Qadv=0.0     [before step]
                20060702 02:00  t=7200 s    Qadv=-8.0e-8 [after step: drying]
                20060702 05:00  t=18000 s   Qadv=-8.0e-8 [before step]
                20060702 05:00  t=18000 s   Qadv=0.0     [after step]
                20060702 12:00  t=43200 s   Qadv=0.0

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
# Vertical profile function
# Constant at F_col for z >= 200 m; linear taper to 0 below 200 m.
# ============================================================

z_taper = 200.0   # m

def vertical_profile(F_col):
    """Map a column scalar (m/s^2 or K/s) to (nz,) with taper below 200 m."""
    return F_col * np.minimum(z / z_taper, 1.0)

# ============================================================
# Target time axis
# ============================================================

t_tgt = np.arange(nsteps + 1) * dt    # (nsteps+1,) s

# ============================================================
# Table 5: Uadv, Vadv [m/s^2]
# Step change at t=10800 s (03 UTC): 5e-4 -> 0.0
# eps offsets the "after-step" entry by 1 ms to prevent duplicate x in interp.
# ============================================================

_eps = 1e-3   # 1 ms: "before" value placed at t_jump - eps, "after" at t_jump

t_U   = np.array([-3600.0, 10800.0 - _eps, 10800.0, 43200.0])
Ua_t  = np.array([ 5.0e-4,  5.0e-4,        0.0,     0.0    ])  # Uadv column value
Va_t  = np.array([ 0.0,     0.0,            0.0,     0.0    ])  # Vadv column value (always 0)

Uadv_col = np.interp(t_tgt, t_U, Ua_t)   # (nsteps+1,)
Vadv_col = np.interp(t_tgt, t_U, Va_t)   # (nsteps+1,)

# ============================================================
# Table 6: THadv [K/s]
# Step changes at t=3600 s (01 UTC) and t=21600 s (06 UTC)
# ============================================================

t_TH   = np.array([-43200.0, 3600.0 - _eps, 3600.0, 21600.0 - _eps, 21600.0, 43200.0])
THa_t  = np.array([ -2.5e-5, -2.5e-5,       7.5e-5,  7.5e-5,        0.0,     0.0    ])

THadv_col = np.interp(t_tgt, t_TH, THa_t)   # (nsteps+1,)

# ============================================================
# Table 7: Qadv [kg/kg/s]
# Step changes at t=7200 s (02 UTC) and t=18000 s (05 UTC)
# ============================================================

t_Q   = np.array([0.0,  7200.0 - _eps, 7200.0, 18000.0 - _eps, 18000.0, 43200.0])
Qa_t  = np.array([0.0,  0.0,          -8.0e-8, -8.0e-8,        0.0,     0.0    ])

Qadv_col = np.interp(t_tgt, t_Q, Qa_t)    # (nsteps+1,)

# ============================================================
# Assemble 2D series (nsteps+1, nz) with vertical taper
# ============================================================

Uadv_series  = np.array([vertical_profile(v) for v in Uadv_col])   # (nsteps+1, nz)
Vadv_series  = np.array([vertical_profile(v) for v in Vadv_col])
THadv_series = np.array([vertical_profile(v) for v in THadv_col])
Qadv_series  = np.array([vertical_profile(v) for v in Qadv_col])

# ============================================================
# Save
# ============================================================

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

out_path = os.path.join(input_dir, 'AdvForcing.npz')
np.savez(out_path,
         Uadv_series  = Uadv_series.astype(np.float64),
         Vadv_series  = Vadv_series.astype(np.float64),
         THadv_series = THadv_series.astype(np.float64),
         Qadv_series  = Qadv_series.astype(np.float64),
         dt_adv       = np.float64(dt),
         optAdvection = np.int32(1))

print(f"AdvForcing file written to {out_path}")
print(f"  nsteps+1 = {nsteps + 1}  (dt = {dt} s, SimTime = {SimTime} s)")
print(f"  nz = {nz},  dz = {dz:.4f} m")
_i3h  = int(3*3600/dt)
_i1h  = int(1*3600/dt)
_i6h  = int(6*3600/dt)
_i2h  = int(2*3600/dt)
_i5h  = int(5*3600/dt)
print(f"  Uadv column at t=0h      : {Uadv_col[0]:.3e} m/s^2  (5e-4)")
print(f"  Uadv column at t=3h-dt   : {Uadv_col[_i3h - 1]:.3e} m/s^2  (before step)")
print(f"  Uadv column at t=3h+dt   : {Uadv_col[_i3h + 1]:.3e} m/s^2  (after step -> 0)")
print(f"  THadv column at t=0h     : {THadv_col[0]:.3e} K/s  (-2.5e-5)")
print(f"  THadv column at t=1h-dt  : {THadv_col[_i1h - 1]:.3e} K/s  (before step)")
print(f"  THadv column at t=1h+dt  : {THadv_col[_i1h + 1]:.3e} K/s  (after step -> +7.5e-5)")
print(f"  THadv column at t=6h-dt  : {THadv_col[_i6h - 1]:.3e} K/s  (before step)")
print(f"  THadv column at t=6h+dt  : {THadv_col[_i6h + 1]:.3e} K/s  (after step -> 0)")
print(f"  Qadv column at t=2h-dt   : {Qadv_col[_i2h - 1]:.3e} kg/kg/s  (before step)")
print(f"  Qadv column at t=2h+dt   : {Qadv_col[_i2h + 1]:.3e} kg/kg/s  (after step -> -8e-8)")
print(f"  Qadv column at t=5h-dt   : {Qadv_col[_i5h - 1]:.3e} kg/kg/s  (before step)")
print(f"  Qadv column at t=5h+dt   : {Qadv_col[_i5h + 1]:.3e} kg/kg/s  (after step -> 0)")
print(f"  Uadv_series  shape: {Uadv_series.shape}")
print(f"  THadv_series shape: {THadv_series.shape}")
print(f"  Qadv_series  shape: {Qadv_series.shape}")
