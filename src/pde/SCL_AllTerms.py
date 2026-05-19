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
File: SCL_AllTerms.py
======================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: computes all the right hand side terms of scalar equations
"""

# ============================================================
#  Imports
# ============================================================

import jax

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *

# Import utility functions
from ..utilities.Utilities import PlanarMean


# ============================================================
# Right hand side terms for scalar equations
# ============================================================

@jax.jit
def RHS_Scalar(TH, THAdvectionSum, divq, RayleighDampCoeff_stag, THadv):
    """
    Computes the right-hand side terms for the scalar transport equation.

    Parameters:
    -----------
    THAdvectionSum : ndarray of shape (nx, ny, nz)
        Sum of advection terms for potential temperature
    divq : ndarray of shape (nx, ny, nz)
        Divergence of subgrid-scale scalar flux
    THadv : ndarray of shape (nx, ny, nz)
        Non-dimensional large-scale (mesoscale) advection tendency for potential
        temperature.  Pass ZeRo3D when optAdvection == 0.

    Returns:
    --------
    RHS_TH : ndarray of shape (nx, ny, nz)
        Right-hand side terms for potential temperature equation
    """

    RHS_TH = - THAdvectionSum - divq

    if optAdvection >= 1:
        RHS_TH = RHS_TH + THadv

    if optDamping == 1:
        # Compute mean potential temperature at each vertical level
        TH_bar = PlanarMean(TH)

        # Compute temperature fluctuations from mean
        TH_fluc = TH - TH_bar.reshape(1, 1, -1)

        RHS_TH = RHS_TH - RayleighDampCoeff_stag * TH_fluc

    return RHS_TH


@jax.jit
def RHS_Moisture(Q, QAdvectionSum, divqm, RayleighDampCoeff_stag, Qadv):
    """
    Right-hand side for the specific humidity (Q) transport equation.

    Parameters:
    -----------
    Q : ndarray (nx, ny, nz) — specific humidity (kg/kg)
    QAdvectionSum : ndarray (nx, ny, nz) — advection sum for Q
    divqm : ndarray (nx, ny, nz) — divergence of SGS moisture flux
    RayleighDampCoeff_stag : ndarray (nx, ny, nz) — Rayleigh damping coefficients
    Qadv : ndarray (nx, ny, nz) — large-scale moisture advection tendency
                                   (ZeRo3D when no explicit forcing is prescribed)

    Returns:
    --------
    RHS_Q : ndarray (nx, ny, nz)
    """
    RHS_Q = - QAdvectionSum - divqm + Qadv

    if optDamping == 1:
        Q_bar  = PlanarMean(Q)
        Q_fluc = Q - Q_bar.reshape(1, 1, -1)
        RHS_Q  = RHS_Q - RayleighDampCoeff_stag * Q_fluc

    return RHS_Q
