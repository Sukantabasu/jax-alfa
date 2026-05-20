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

File: NSE_BuoyancyTerms.py
==========================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-4
:Description: computes the buoyancy term for use in the momentum equation
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.ConfigLoader import *

# Import derived variables
from ..config.DerivedVars import *

# Import utility functions
from ..utilities.Utilities import PlanarMean


# ============================================================
#  Compute buoyancy terms using reference temperature (Option 0)
# ============================================================

@jax.jit
def BuoyancyOpt1(TH, H, ZeRo3D):
    """
    Computes the buoyancy term using reference temperature.
    buoyancy: g*(Tv - <Tv>)/T_0

    Parameters:
    -----------
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature (normalized)
    H : ndarray of shape (nx, ny, nz), optional
        Specific humidity (normalized)
    ZeRo3D : ndarray of shape (nx, ny, nz)
        Pre-allocated zero array

    Returns:
    --------
    buoyancy : ndarray of shape (nx, ny, nz)
        Buoyancy term for momentum equation
    """

    # Initialize arrays with zeros
    buoyancy = ZeRo3D.copy()

    # TH is stored as anomaly (TH - T_0); use absolute temperature so that
    # the moisture term 0.61*Q*T_0 is O(~2 K) not O(~0 K).
    THv = (TH + T_0_nondim) * (1 + 0.61 * H)

    # Compute planar-averaged virtual potential temperature
    THv_bar = PlanarMean(THv)

    # Reshape THv_bar for broadcasting
    THv_bar_3D = THv_bar.reshape(1, 1, -1)

    # Compute normalized deviation
    THv_normalized = (THv - THv_bar_3D) / T_0_nondim

    # Compute the half-level averages for all levels
    above = THv_normalized[:, :, 1:nz]
    below = THv_normalized[:, :, 0:nz - 1]
    buoyancy_interior = 0.5 * g_nondim * (above + below)

    # Update the buoyancy array
    buoyancy = buoyancy.at[:, :, 1:nz].set(buoyancy_interior)

    # Set bottom boundary condition
    buoyancy = buoyancy.at[:, :, 0].set(0.0)

    return buoyancy


# ============================================================
#  Compute buoyancy terms using local mean temperature (Option 1)
# ============================================================

@jax.jit
def BuoyancyOpt2(TH, H, ZeRo3D):
    """
    Computes the buoyancy term using local mean virtual
    potential temperature.
    buoyancy: g*(Tv - <Tv>)/<Tv>

    Parameters:
    -----------
    TH : ndarray of shape (nx, ny, nz)
        Potential temperature (normalized)
    H : ndarray of shape (nx, ny, nz), optional
        Specific humidity (normalized)

    Returns:
    --------
    buoyancy : ndarray of shape (nx, ny, nz)
        Buoyancy term for momentum equation
    """

    # Initialize arrays with zeros
    buoyancy = ZeRo3D.copy()

    # TH is stored as anomaly (TH - T_0); use absolute temperature so that
    # the moisture term 0.61*Q*T_0 is O(~2 K) not O(~0 K).
    THv = (TH + T_0_nondim) * (1 + 0.61 * H)

    # Compute plane-averaged virtual potential temperature
    THv_bar = PlanarMean(THv)

    # Reshape THv_bar for broadcasting
    THv_bar_3D = THv_bar.reshape(1, 1, -1)

    # THv_bar is already ~T_0_nondim*(1+0.61*<Q>), so use it directly.
    THv_normalized = (THv - THv_bar_3D) / THv_bar_3D

    # Compute the half-level averages for all levels
    above = THv_normalized[:, :, 1:nz]
    below = THv_normalized[:, :, 0:nz - 1]
    buoyancy_interior = 0.5 * g_nondim * (above + below)

    # Update the buoyancy array
    buoyancy = buoyancy.at[:, :, 1:nz].set(buoyancy_interior)

    # Set bottom boundary condition
    buoyancy = buoyancy.at[:, :, 0].set(0.0)

    return buoyancy
