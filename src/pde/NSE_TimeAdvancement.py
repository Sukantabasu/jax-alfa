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
File: NSE_TimeAdvancement.py
===============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: implements Adams-Bashforth (AB2) for time integration
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import configuration from namelist
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *


# ============================================================
#  Time advancement using Adams-Bashforth scheme
# ============================================================

@jax.jit
def AB2_uvw(u, v, w,
            RHS_u, RHS_u_previous,
            RHS_v, RHS_v_previous,
            RHS_w, RHS_w_previous):
    """
    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Current velocity components
    RHS_u, RHS_v, RHS_w : ndarray of shape (nx, ny, nz)
        Current right-hand side terms for velocity components
    RHS_u_previous, RHS_v_previous, RHS_w_previous :
        ndarray of shape (nx, ny, nz)
        Previous right-hand side terms for velocity components

    Returns:
    --------
    u_new, v_new, w_new : ndarray of shape (nx, ny, nz)
        Updated velocity components after time advancement
    """

    u_new = u + dt_nondim * (1.5 * RHS_u - 0.5 * RHS_u_previous)
    v_new = v + dt_nondim * (1.5 * RHS_v - 0.5 * RHS_v_previous)
    w_new = w + dt_nondim * (1.5 * RHS_w - 0.5 * RHS_w_previous)

    # Apply boundary conditions
    u_new = u_new.at[:, :, nz - 1].set(u_new[:, :, nz - 2])
    v_new = v_new.at[:, :, nz - 1].set(v_new[:, :, nz - 2])
    w_new = w_new.at[:, :, nz - 1].set(0.0)
    w_new = w_new.at[:, :, 0].set(0.0)

    return u_new, v_new, w_new
