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
File: NSE_AllTerms.py
========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: computes all the right hand side terms of NSE
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import configuration from namelist
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *

# Import utility functions
from ..utilities.Utilities import PlanarMean


# ============================================================
# Right hand side terms for momentum equations
# ============================================================

@jax.jit
def RHS_Momentum(u, v, w,
                 Ug, Vg,
                 Cx, Cy, Cz,
                 buoyancy,
                 divtx, divty, divtz,
                 RayleighDampCoeff, RayleighDampCoeff_stag):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components in x, y, and z directions
    Ug, Vg : ndarray of shape (nx, ny, nz)
        Geostrophic winds in x and y directions
    Cx, Cy, Cz : ndarray of shape (nx, ny, nz)
        Advection terms
    buoyancy : ndarray of shape (nx, ny, nz)
        Buoyancy term
    divtx, divty, divtz : ndarray of shape (nx, ny, nz)
        SGS stress divergence terms

    Returns:
    --------
    RHS_u : ndarray of shape (nx, ny, nz)
        RHS for u component
    RHS_v : ndarray of shape (nx, ny, nz)
        RHS for v component
    RHS_w : ndarray of shape (nx, ny, nz)
        RHS for w component
    """

    RHS_u = - Cx - divtx - f_coriolis_nondim * (Vg - v)
    RHS_v = - Cy - divty + f_coriolis_nondim * (Ug - Ugal - u)
    RHS_w = - Cz - divtz + buoyancy

    if optDamping == 1:

        # Compute mean velocity components at each vertical level
        u_bar = PlanarMean(u)
        v_bar = PlanarMean(v)
        w_bar = PlanarMean(w)

        # Compute velocity fluctuations from mean
        u_fluc = u - u_bar.reshape(1, 1, -1)
        v_fluc = v - v_bar.reshape(1, 1, -1)
        w_fluc = w - w_bar.reshape(1, 1, -1)

        RHS_u = RHS_u - RayleighDampCoeff_stag * u_fluc
        RHS_v = RHS_v - RayleighDampCoeff_stag * v_fluc
        RHS_w = RHS_w - RayleighDampCoeff * w_fluc

    return RHS_u, RHS_v, RHS_w
