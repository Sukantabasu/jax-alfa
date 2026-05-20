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
File: CreateInputs_GABLS1.py
================================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-20
:Description: Creates vel.ini and TH.ini for the SBL_GABLS1 case.
              Reads grid parameters from Config.py in the same directory.

              Initial conditions after Beare et al. (2006):
                u = Ug = 8 m/s, v = 0 m/s
                TH = 265 K for z <= 100 m
                TH = 265 + 0.01*(z - 100) K for z > 100 m
              Small random perturbations below 50 m.
"""

import numpy as np
import os

np.random.seed(42)

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

Ug         = 8.0
T_sfc_init = 265.0
inversion  = 0.01  # K/m above 100 m
rw = 0.01
rt = 0.01

u  = Ug * np.ones((nx, ny, nz))
v  = np.zeros((nx, ny, nz))
w  = np.zeros((nx, ny, nz))
TH = np.zeros((nx, ny, nz))

for k in range(nz):
    TH[:, :, k] = T_sfc_init if z[k] <= 100.0 else T_sfc_init + inversion * (z[k] - 100.0)
    if z[k] <= 50.0:
        w[:, :, k]  += rw * np.random.randn(nx, ny)
        TH[:, :, k] += rt  * np.random.randn(nx, ny)

u_flat  = u.reshape(-1, order='F')
v_flat  = v.reshape(-1, order='F')
w_flat  = w.reshape(-1, order='F')
TH_flat = TH.reshape(-1, order='F')
vel_data = np.column_stack([u_flat, v_flat, w_flat])

input_dir = os.path.join(_script_dir, 'input')
os.makedirs(input_dir, exist_ok=True)
np.savetxt(os.path.join(input_dir, 'vel.ini'), vel_data)
np.savetxt(os.path.join(input_dir, 'TH.ini'),  TH_flat)

print(f"GABLS1 initial conditions written to {{input_dir}}")
print(f"  Grid: nx={{nx}}, ny={{ny}}, nz={{nz}}, dz={{dz:.4f}} m")
print(f"  vel.ini shape: {{vel_data.shape}}")
print(f"  TH.ini  shape: {{TH_flat.shape}}")
print(f"  TH range: {{TH.min():.3f}} - {{TH.max():.3f}} K")
