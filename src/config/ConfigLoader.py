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
File: ConfigLoader.py
=====================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2026-05-01
:Description: Loads run-specific configuration from JAXALFA_RUNDIR/Config.py.
              JAXALFA_RUNDIR must be set in the environment before running.

Usage
-----
Before launching JAX-ALFA, set the run directory::

    export JAXALFA_RUNDIR=/path/to/run_directory
    python -m src.Main

The run directory must contain a Config.py with all simulation parameters.
"""


# ============================================================
# Imports
# ============================================================

import os
import numpy as np


# ============================================================
# Load run-directory configuration (mandatory)
# ============================================================

_rundir = os.environ.get('JAXALFA_RUNDIR')
if _rundir is None:
    raise EnvironmentError(
        "\n\nJAXALFA_RUNDIR is not set.\n"
        "Set it to the run directory before launching JAX-ALFA:\n\n"
        "    export JAXALFA_RUNDIR=/path/to/run_directory\n"
        "    python $JAXALFA_RUNDIR/CreateInputs*.py\n"
        "    python -m src.Main\n"
    )

_config_path = os.path.join(_rundir, 'Config.py')
if not os.path.isfile(_config_path):
    raise FileNotFoundError(
        f"\nConfig.py not found in JAXALFA_RUNDIR.\n"
        f"Expected: {_config_path}\n"
    )

with open(_config_path) as _f:
    exec(_f.read())

# Backward-compatible defaults for time-varying surface BC
if 'optSurfBC' not in dir():
    optSurfBC = 0
if 'SurfaceBCFile' not in dir():
    SurfaceBCFile = 'input/SurfaceBC.npz'

# Backward-compatible defaults for time/height-varying geostrophic wind
# optGeoWind = 0: constant Ug2, Vg2 from Config (default, all existing cases)
# optGeoWind = 1: time + height varying, loaded from GeoWindFile
if 'optGeoWind' not in dir():
    optGeoWind = 0
if 'GeoWindFile' not in dir():
    GeoWindFile = 'input/GeoWind.npz'

# Backward-compatible default for screen-level temperature reference height.
# zTemperature = 0  : use z0T as the thermal reference height (default, GABLS1/SBL)
# zTemperature > 0  : use this height (m) as the reference for the heat-flux
#                     denominator and psi_h0 — needed when the prescribed surface
#                     temperature is observed at a screen height above z0T (e.g.
#                     Wangara: temperature measured at 1.2 m above ground).
if 'zTemperature' not in dir():
    zTemperature = 0.0

# Backward-compatible defaults for time/height-varying large-scale advection.
# optAdvection = 0: no mesoscale advection forcing (default, all existing cases)
# optAdvection = 1: time + height varying, loaded from AdvectionFile
if 'optAdvection' not in dir():
    optAdvection = 0
if 'AdvectionFile' not in dir():
    AdvectionFile = 'input/AdvForcing.npz'

# Backward-compatible defaults for moisture.
# optMoisture = 0: no moisture (default, all existing cases)
# optMoisture = 1: specific humidity Q is a prognostic variable
if 'optMoisture' not in dir():
    optMoisture = 0
# zMoisture: screen-level reference height for moisture flux denominator (m).
#   0  : use z0T as the moisture reference height (same as heat)
#   > 0: use this height (e.g. 0.25 m when surface Q observed at 0.25 m)
if 'zMoisture' not in dir():
    zMoisture = 0.0
# MoistureFlux: constant surface moisture flux (kg/kg m/s), used when optMoistureSurfBC=0
if 'MoistureFlux' not in dir():
    MoistureFlux = 0.0
# optMoistureSurfBC: 0 = constant flux (MoistureFlux)
#                   1 = time-varying flux (from MoistureSurfaceBCFile)
#                   2 = time-varying surface Q (from MoistureSurfaceBCFile)
if 'optMoistureSurfBC' not in dir():
    optMoistureSurfBC = 0
if 'MoistureSurfaceBCFile' not in dir():
    MoistureSurfaceBCFile = 'input/MoistureSurfaceBC.npz'
# q_inversion: specific humidity lapse rate above domain top (kg/kg/m).
#   0 : zero gradient (flat Q profile at top, default)
if 'q_inversion' not in dir():
    q_inversion = 0.0

# GPU_ID: which GPU to use when optGPU=1 (0-indexed).
# Set to 1 in Config.py to run on the second GPU.
if 'GPU_ID' not in dir():
    GPU_ID = 0

# Backward-compatible defaults for the stability-dependent Smagorinsky model
# (STAB-SM, optSgs=5).  Override any of these in Config.py if needed.
# CsMO_SM : Smagorinsky coefficient (0.23 recommended for finite-difference codes)
# aMO_SM  : heat stability parameter (= 1/Pr_t_neutral)
# bMO_SM, cMO_SM : unstable stability function coefficients
# fMO_SM, gMO_SM, hMO_SM : stable stability function coefficients
# RicMO_SM : critical Richardson number
# rMO_SM   : stable stability function exponent
if 'CsMO_SM'  not in dir(): CsMO_SM  = 0.17
if 'aMO_SM'   not in dir(): aMO_SM   = 1.0 / 0.7
if 'bMO_SM'   not in dir(): bMO_SM   = 40.0
if 'cMO_SM'   not in dir(): cMO_SM   = 16.0
if 'fMO_SM'   not in dir(): fMO_SM   = 1.0 / 0.7
if 'gMO_SM'   not in dir(): gMO_SM   = 1.2
if 'hMO_SM'   not in dir(): hMO_SM   = 0.0
if 'RicMO_SM' not in dir(): RicMO_SM = 0.25
if 'rMO_SM'   not in dir(): rMO_SM   = 4.0

# Backward-compatible default for float precision
if 'use_double_precision' not in dir():
    use_double_precision = False

# Pressure solver selection.
# optPressureSolver = 0: LU    — dense matrix + jnp.linalg.solve (original)
# optPressureSolver = 1: Thomas — tridiagonal Thomas algorithm (faster)
if 'optPressureSolver' not in dir():
    optPressureSolver = 0

del _rundir, _config_path, _f
