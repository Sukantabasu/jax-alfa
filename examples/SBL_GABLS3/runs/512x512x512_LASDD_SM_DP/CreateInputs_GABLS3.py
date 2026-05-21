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
File: CreateInputs_GABLS3.py
==============================

:Author: Sukanta Basu
:Date: 2026-05-19
:Description: Creates vel.npy, TH.npy, and Q.npy for the GABLS3 case.

              Initial conditions from the 00 UTC 2 July 2006 sounding
              (Tables 2 and 3 of GABLS3_LES_Revised.docx):
                Table 2: z(m), U(m/s), V(m/s)
                Table 3: z(m), P(hPa), theta(K), Q(kg/kg)

              Profiles are linearly interpolated onto model half-levels;
              scipy extrapolates outside the sounding range.

              Random perturbations (from GABLS3_LES_July2_00UTC_Initialization.m):
                u, v : variance Ru = 0.2*(1-z/200)^2  =>  sigma = sqrt(0.2)*(1-z/200)
                       ~0.447 m/s at z=0, tapering linearly to 0 at z=200 m
                TH   : amplitude 0.1 K (constant), applied for z <= 200 m

              Output files (written to input/ sub-directory):
                vel.npy   — dimensional (u, v, w) in m/s, Fortran order
                TH.npy    — dimensional potential temperature in K, Fortran order
                Q.npy     — dimensional specific humidity in kg/kg, Fortran order

              Place this script in the run directory alongside Config.py.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import os
from scipy.interpolate import interp1d

np.random.seed(42)

# ============================================================
# Read grid parameters from Config.py
# ============================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

nx  = int(_cfg['nx'])
ny  = int(_cfg['ny'])
nz  = int(_cfg['nz'])
l_z = float(_cfg['l_z'])

# ============================================================
# Vertical grid — half levels (u, v, TH)
# z_k = (k + 0.5) * dz,  dz = l_z / (nz - 1)
# ============================================================

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])

# ============================================================
# Initial velocity profile (Table 2)
# ============================================================

z_vel = np.array([10, 20, 40, 80, 140, 200, 203, 257, 308, 310,
                  363, 408, 426, 465, 520, 541, 575, 635, 657,
                  694, 715, 749, 772, 801, 830], dtype=float)

U_obs = np.array([-3.35, -4.31, -6.13, -9.00, -11.48, -10.16,
                  -10.08, -9.15, -8.75, -8.74, -8.57, -8.46,
                  -8.41, -8.23, -7.82, -7.60, -7.17, -6.22,
                  -5.82, -5.14, -4.77, -4.19, -3.86, -3.54,
                  -3.37], dtype=float)

V_obs = np.array([-0.04, 0.08, 0.32, 0.90, 3.50, 5.58,
                   5.57, 5.47, 5.54, 5.55, 5.64, 5.64,
                   5.61, 5.47, 5.10, 4.90, 4.53, 3.76,
                   3.47, 3.00, 2.76, 2.46, 2.34, 2.31,
                   2.46], dtype=float)

# ============================================================
# Initial potential temperature profile (Table 3)
# ============================================================

z_th = np.array([10, 20, 40, 80, 140, 200, 203, 257, 308,
                 363, 408, 465, 520, 575, 635, 694, 749,
                 801, 854], dtype=float)

TH_obs = np.array([292.72, 293.02, 293.41, 294.30, 295.68, 297.35,
                   297.35, 297.51, 297.66, 297.81, 297.94, 298.11,
                   298.27, 298.42, 298.60, 298.77, 298.93, 299.08,
                   299.23], dtype=float)

# ============================================================
# Initial specific humidity profile (Table 3)
# Same z-levels as temperature sounding
# ============================================================

Q_obs = np.array([0.0098, 0.0097, 0.0096, 0.0096, 0.0092, 0.0089,
                  0.0089, 0.0089, 0.0089, 0.0089, 0.0089, 0.0089,
                  0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088,
                  0.0087], dtype=float)

# ============================================================
# Interpolate onto model half-levels
# ============================================================

_interp_U  = interp1d(z_vel, U_obs,  kind='linear', fill_value='extrapolate')
_interp_V  = interp1d(z_vel, V_obs,  kind='linear', fill_value='extrapolate')
_interp_TH = interp1d(z_th,  TH_obs, kind='linear', fill_value='extrapolate')
_interp_Q  = interp1d(z_th,  Q_obs,  kind='linear', fill_value='extrapolate')

U1D  = _interp_U(z)
V1D  = _interp_V(z)
TH1D = _interp_TH(z)
Q1D  = _interp_Q(z)

# ============================================================
# Build 3D arrays and add random perturbations for z <= 200 m
# ============================================================

u  = np.zeros((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = np.zeros((nx, ny, nz))
Q  = np.zeros((nx, ny, nz))

for k in range(nz):
    u[:, :, k]  = U1D[k]
    v[:, :, k]  = V1D[k]
    TH[:, :, k] = TH1D[k]
    Q[:, :, k]  = Q1D[k]

for k in range(nz):
    if z[k] <= 200.0:
        sigma_uv = np.sqrt(0.2) * (1.0 - z[k] / 200.0)
        u[:, :, k]  += sigma_uv * np.random.randn(nx, ny)
        v[:, :, k]  += sigma_uv * np.random.randn(nx, ny)
        TH[:, :, k] += 0.1 * np.random.randn(nx, ny)

# ============================================================
# Flatten (Fortran order) and save
# ============================================================

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')
Q_flat  = Q.reshape(-1, order='F')

vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)

np.save(os.path.join(input_dir, 'vel.npy'), vel_data)
np.save(os.path.join(input_dir, 'TH.npy'),  TH_flat)
np.save(os.path.join(input_dir, 'Q.npy'),   Q_flat)

print(f"GABLS3 initial conditions written to {input_dir}")
print(f"  vel.npy shape: {vel_data.shape}  (u, v, w columns in m/s)")
print(f"  TH.npy  shape: {TH_flat.shape}")
print(f"  Q.npy   shape: {Q_flat.shape}")
print(f"  Grid: nz={nz}, dz={dz:.4f} m, z[0]={z[0]:.4f} m, z[-1]={z[-1]:.4f} m")
print(f"  U range : [{U1D.min():.3f}, {U1D.max():.3f}] m/s")
print(f"  V range : [{V1D.min():.3f}, {V1D.max():.3f}] m/s")
print(f"  TH range: [{TH1D.min():.3f}, {TH1D.max():.3f}] K")
print(f"  Q range : [{Q1D.min():.5f}, {Q1D.max():.5f}] kg/kg")
