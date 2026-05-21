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
File: CreateInputs_Wangara.py
==============================

:Author: Sukanta Basu
:Date: 2026-05-18
:Description: Creates vel.npy and TH.npy for the Wangara diurnal cycle case
              (Basu et al. 2008, BLM).

              Simulation period: full diurnal cycle from 0900 LST day 33
              (16 August 1967) to 0900 LST day 34 (17 August 1967).

              Initial conditions are interpolated from the day 33, 0900 LST
              sounding (Wangara_Sounding3309.txt):
                Columns: z(m), P(mb), T(C), U(m/s), V(m/s)
                Theta = (T + 273.16) + z * 10/1000   [Mellor-Yamada DALR]

              Random perturbations (matching Basu et al. 2008):
                u : amplitude 1e-2 m/s, applied below z = 100 m
                TH: amplitude 0.10  K, applied below z = 100 m

              Output files (written to input/ sub-directory):
                vel.npy   — dimensional (u, v, w) in m/s, Fortran order
                TH.npy    — dimensional potential temperature in K, Fortran order

              Place this script in the run directory alongside Config.py.
              Set SOUNDING_FILE below to the path of Wangara_Sounding3309.txt.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os
from scipy.interpolate import interp1d

np.random.seed(42)

# ============================================================
# Path to sounding data
# (adjust if the file lives elsewhere)
# ============================================================

_script_dir  = os.path.dirname(os.path.abspath(__file__))
SOUNDING_FILE = os.path.join(_script_dir, 'Wangara_Sounding3309.txt')

# ============================================================
# Read grid parameters from Config.py (same directory)
# ============================================================

_cfg = {}
_cfg_path = os.path.join(_script_dir, 'Config.py')
with open(_cfg_path) as _f:
    exec(_f.read(), _cfg)

nx  = int(_cfg['nx'])
ny  = int(_cfg['ny'])
nz  = int(_cfg['nz'])
l_z = float(_cfg['l_z'])   # domain height (m)

# ============================================================
# Vertical grid — half levels (u, v, TH)
# z_k = (k + 0.5) * dz,  dz = l_z / (nz - 1)
# ============================================================

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])

# ============================================================
# Load and interpolate sounding
# ============================================================

if not os.path.isfile(SOUNDING_FILE):
    raise FileNotFoundError(
        f"Sounding file not found: {SOUNDING_FILE}\n"
        "Provide Wangara_Sounding3309.txt (columns: z, P, T, U, V)."
    )

D    = np.loadtxt(SOUNDING_FILE)
zob  = D[:, 0]   # height (m)
Tob  = D[:, 2]   # temperature (C)
Uob  = D[:, 3]   # zonal wind (m/s)
Vob  = D[:, 4]   # meridional wind (m/s)

# Potential temperature: Mellor-Yamada DALR correction
Thetaob = (Tob + 273.16) + zob * 10.0 / 1000.0

# Interpolate onto model half-levels (linear, with extrapolation)
_interp = lambda obs: interp1d(zob, obs, kind='linear', fill_value='extrapolate')
U1D  = _interp(Uob)(z)
V1D  = _interp(Vob)(z)
TH1D = _interp(Thetaob)(z)

# ============================================================
# Build 3D arrays and add random perturbations below 100 m
# ============================================================

u  = np.zeros((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = np.zeros((nx, ny, nz))

for k in range(nz):
    u[:, :, k]  = U1D[k]
    v[:, :, k]  = V1D[k]
    TH[:, :, k] = TH1D[k]

for k in range(nz):
    if z[k] <= 100.0:
        u[:, :, k]  += 1e-2 * np.random.randn(nx, ny)
        TH[:, :, k] += 0.10 * np.random.randn(nx, ny)

# ============================================================
# Flatten (Fortran order: i varies fastest, then j, then k)
# and save
# ============================================================

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')

vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

np.save(os.path.join(input_dir, 'vel.npy'), vel_data)
np.save(os.path.join(input_dir, 'TH.npy'),  TH_flat)

print(f"Wangara initial conditions written to {input_dir}")
print(f"  vel.npy  shape: {vel_data.shape}  (u, v, w columns in m/s)")
print(f"  TH.npy   shape: {TH_flat.shape}")
print(f"  Grid: nz={nz}, dz={dz:.3f} m, z[0]={z[0]:.3f} m, z[-1]={z[-1]:.3f} m")
print(f"  U range : [{U1D.min():.2f}, {U1D.max():.2f}] m/s")
print(f"  V range : [{V1D.min():.2f}, {V1D.max():.2f}] m/s")
print(f"  TH range: [{TH1D.min():.2f}, {TH1D.max():.2f}] K")
