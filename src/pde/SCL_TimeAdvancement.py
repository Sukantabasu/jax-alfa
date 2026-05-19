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
File: SCL_TimeAdvancement.py
=============================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: implements Adams-Bashforth (AB2) for time integration
of scalar equations
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *


# ============================================================
#  Time advancement for potential temperature
# ============================================================

@jax.jit
def AB2_TH(TH,
           RHS_TH, RHS_TH_previous):
    """
    Parameters:
    -----------
    TH : ndarray of shape (nx, ny, nz)
        Current potential temperature field
    RHS_TH : ndarray of shape (nx, ny, nz)
        Current right-hand side terms for potential temperature
    RHS_TH_previous : ndarray of shape (nx, ny, nz)
        Previous right-hand side terms for potential temperature

    Returns:
    --------
    TH_new : ndarray of shape (nx, ny, nz)
        Updated potential temperature field after time advancement
    """

    TH_new = TH + dt_nondim * (1.5 * RHS_TH - 0.5 * RHS_TH_previous)

    # Boundary condition for top of the domain:
    TH_new = TH_new.at[:, :, nz - 1].set(
        TH_new[:, :, nz - 2] + inversion_nondim * dz
    )

    return TH_new


# ============================================================
#  Time advancement for specific humidity
# ============================================================

@jax.jit
def AB2_Q(Q,
          RHS_Q, RHS_Q_previous):
    """
    Parameters:
    -----------
    Q : ndarray (nx, ny, nz) — current specific humidity (kg/kg)
    RHS_Q : ndarray (nx, ny, nz) — current RHS for Q
    RHS_Q_previous : ndarray (nx, ny, nz) — previous RHS for Q

    Returns:
    --------
    Q_new : ndarray (nx, ny, nz) — updated specific humidity
    """
    Q_new = Q + dt_nondim * (1.5 * RHS_Q - 0.5 * RHS_Q_previous)

    # Top BC: apply free-atmosphere Q gradient (default q_inversion=0: zero gradient)
    Q_new = Q_new.at[:, :, nz - 1].set(
        Q_new[:, :, nz - 2] + q_inversion_nondim * dz
    )

    return Q_new
