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
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
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

del _rundir, _config_path, _f
