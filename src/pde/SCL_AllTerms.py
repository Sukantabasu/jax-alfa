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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-29
:Description: computes all the right hand side terms of scalar equations
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
# Right hand side terms for scalar equations
# ============================================================

@jax.jit
def RHS_Scalar(TH, THAdvectionSum, divq, RayleighDampCoeff_stag):
    """
    Computes the right-hand side terms for the scalar transport equation.

    Parameters:
    -----------
    THAdvectionSum : ndarray of shape (nx, ny, nz)
        Sum of advection terms for potential temperature
    divq : ndarray of shape (nx, ny, nz)
        Divergence of subgrid-scale scalar flux

    Returns:
    --------
    RHS_TH : ndarray of shape (nx, ny, nz)
        Right-hand side terms for potential temperature equation
    """

    RHS_TH = - THAdvectionSum - divq

    if optDamping == 1:
        # Compute mean potential temperature at each vertical level
        TH_bar = PlanarMean(TH)

        # Compute temperature fluctuations from mean
        TH_fluc = TH - TH_bar.reshape(1, 1, -1)

        RHS_TH = RHS_TH - RayleighDampCoeff_stag * TH_fluc

    return RHS_TH
