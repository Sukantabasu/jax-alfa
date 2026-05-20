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
File: CreateInputs_CBL_N91.py
================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-20
:Description: Creates vel.ini and TH.ini for the CBL_N91 case.
              Reads grid parameters from Config.py in the same directory.

              Initial conditions after Nieuwstadt et al. (1991):
                u = 0 m/s, v = 0 m/s
                TH = 300 K for z <= z_i = 1350.4 m
                TH = 300 + 0.003*(z - z_i) K for z > z_i
              Random perturbations (amplitude 0.03) below 200 m.
"""

import numpy as np
import os

np.random.seed(0)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cfg = {{}}
with open(os.path.join(_script_dir, 'Config.py')) as _f:
    exec(_f.read(), _cfg)

nx  = int(_cfg['nx'])
ny  = int(_cfg['ny'])
nz  = int(_cfg['nz'])
l_z = float(_cfg['l_z'])

dz = l_z / (nz - 1)
z  = np.array([(k + 0.5) * dz for k in range(nz)])

Ug        = 0.0
z_i       = 1350.4   # initial boundary layer height (m)
inversion = 3.0/1000 # K/m
T_0       = 300.0
rw = 0.03
rt = 0.03

u  = Ug * np.ones((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = T_0 * np.ones((nx, ny, nz))

for k in range(nz):
    if z[k] <= 200.0:
        r1 = rw * np.random.randn(nx, ny)
        r2 = rt * np.random.randn(nx, ny)
        u[:, :, k]  += r1
        w[:, :, k]  += r1
        TH[:, :, k] += r2
    if z[k] >= z_i:
        TH[:, :, k] += (z[k] - z_i) * inversion

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')
vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)
np.savetxt(os.path.join(input_dir, 'vel.ini'), vel_data)
np.savetxt(os.path.join(input_dir, 'TH.ini'),  TH_flat)

print(f"CBL_N91 initial conditions written to {{input_dir}}")
print(f"  Grid: nx={{nx}}, ny={{ny}}, nz={{nz}}, dz={{dz:.3f}} m")
print(f"  vel.ini shape: {{vel_data.shape}}")
print(f"  TH.ini  shape: {{TH_flat.shape}}")
