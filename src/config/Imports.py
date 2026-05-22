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
File: Imports.py
=======================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: imports all the modules for JAX-ALFA
"""

import os

# Import configurations (namelists)
from . import ConfigLoader as Config

if Config.optGPU == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    print("JAX environment configured for CPU")
else:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Local run: honour GPU_ID from Config
        os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.GPU_ID)
        print(f"JAX environment configured for GPU {Config.GPU_ID}")
    else:
        # Cluster run: SLURM already set CUDA_VISIBLE_DEVICES; don't override
        print(f"JAX environment configured for GPU "
              f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

import jax

if Config.use_double_precision:
    jax.config.update("jax_enable_x64", True)

# Uncomment this line if the dynamic RAM allocation is needed
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def ImportLES():
    # Get the caller's global namespace
    import inspect
    caller_globals = inspect.currentframe().f_back.f_globals

    # Basic libraries
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp

    # Import derived variables module
    from . import DerivedVars

    # Imports from initialization
    from ..initialization.Initialization import Initialize_uvw, Initialize_TH
    from ..initialization.Initialization import Initialize_Q
    from ..initialization.Initialization import Initialize_MoistureSurfaceBC
    from ..initialization.Initialization import Initialize_GeoWind
    from ..initialization.Initialization import Initialize_GeoWind_Varying
    from ..initialization.Initialization import Initialize_RayleighDampingLayer
    from ..initialization.Initialization import Initialize_SurfaceBC
    from ..initialization.Initialization import Initialize_AdvForcing
    from ..initialization.Preprocess import Wavenumber, Constant
    from ..initialization.Preprocess import ZeRo3DIni, ZeRo2DIni, ZeRo1DIni
    from ..initialization.Preprocess import ZeRo3D_padIni, ZeRo3D_fftIni, ZeRo3D_pad_fftIni

    # Imports from operations
    from ..operations.Derivatives import Derivxy, Derivz_M, velocityGradients
    from ..operations.Derivatives import Derivz_TH, potentialTemperatureGradients
    from ..operations.Derivatives import moistureGradients
    from ..operations.Derivatives import Derivz_Generic_uvp, Derivz_Generic_w
    from ..operations.FFT import FFT, FFT_pad
    from ..operations.Filtering import Filtering_Explicit, Filtering_Level1, Filtering_Level2
    from ..operations.Dealiasing import Dealias1, Dealias2

    # Imports from utilities
    from ..utilities.Utilities import StagGridAvg, LogMemory
    from ..utilities.Statistics import ComputeStats, InitializeStats

    # Imports from subgridscale
    from ..subgridscale.StrainRates import StrainsUVPnodes_Dealias, StrainsWnodes_Dealias
    from ..subgridscale.StrainRates import StrainsUVPnodes_NoDealias, StrainsWnodes_NoDealias
    from ..subgridscale.SGSStresses_SM import StressesUVPnodes_Dealias, StressesWnodes_Dealias
    from ..subgridscale.SGSStresses_SM import StressesUVPnodes_NoDealias, StressesWnodes_NoDealias
    from ..subgridscale.SGSStresses_SM import Wall
    from ..subgridscale.DynamicSGS_Main import DynamicSGS, DynamicSGSscalar
    from ..subgridscale.DynamicSGS_LASDD_SM import LASDD as LASDD_SM
    from ..subgridscale.DynamicSGS_LASDD_WL import LASDD as LASDD_WL
    from ..subgridscale.DynamicSGS_ScalarLASDD_SM import ScalarLASDD as ScalarLASDD_SM
    from ..subgridscale.DynamicSGS_ScalarLASDD_WL import ScalarLASDD as ScalarLASDD_WL

    # Get constants from Preprocess
    mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()

    # Select versions based on configuration
    if DerivedVars.optDealias == 1:
        StrainsUVPnodes = StrainsUVPnodes_Dealias
        StrainsWnodes = StrainsWnodes_Dealias
    else:
        StrainsUVPnodes = StrainsUVPnodes_NoDealias
        StrainsWnodes = StrainsWnodes_NoDealias

    # Imports from surface
    from ..surface.SurfaceFlux import MOSTstable, MOSTunstable
    from ..surface.SurfaceFlux import SurfaceFlux_HomogeneousConstantFlux
    from ..surface.SurfaceFlux import SurfaceFlux_HeterogeneousConstantFlux
    from ..surface.SurfaceFlux import SurfaceFlux_HomogeneousVaryingFlux
    from ..surface.SurfaceFlux import SurfaceFlux_HeterogeneousVaryingFlux
    from ..surface.SurfaceFlux import SurfaceFlux_HomogeneousPrescribedTemperature
    from ..surface.SurfaceFlux import SurfaceFlux_HeterogeneousPrescribedTemperature
    from ..surface.SurfaceFlux import SurfaceMoistureFlux_HomogeneousPrescribedQ
    from ..surface.SurfaceFlux import SurfaceMoistureFlux_HeterogeneousPrescribedQ

    # Imports from pde
    from ..pde.NSE_AdvectionTerms import Advection
    from ..pde.SCL_AdvectionTerms import ScalarAdvection
    from ..pde.NSE_BuoyancyTerms import BuoyancyOpt1, BuoyancyOpt2
    from ..pde.NSE_PressureTerms import PressureInit, PressureRC
    from ..pde.NSE_PressureTerms import PressureMatrix, PressureSolve
    from ..pde.NSE_SGSTerms import DivStressStaticSGS, DivStressDynamicSGS
    from ..pde.SCL_SGSTerms import DivFluxStaticSGS, DivFluxDynamicSGS
    from ..pde.NSE_SGSTerms_STABSM import DivStressStaticSGS_STABSM
    from ..pde.SCL_SGSTerms_STABSM import DivFluxStaticSGS_STABSM
    from ..pde.NSE_AllTerms import RHS_Momentum
    from ..pde.SCL_AllTerms import RHS_Scalar, RHS_Moisture
    from ..pde.NSE_TimeAdvancement import AB2_uvw
    from ..pde.SCL_TimeAdvancement import AB2_TH, AB2_Q

    # Add all imports to namespace
    for name, value in locals().items():
        if not name.startswith('_') and name != 'caller_globals' and name != 'inspect':
            caller_globals[name] = value

    # Add all constants from Config to namespace
    for name in dir(Config):
        if not name.startswith('_') and not name.startswith('np'):
            # Add the variable to caller's namespace
            caller_globals[name] = getattr(Config, name)
