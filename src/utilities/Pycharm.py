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
File: Pycharm.py
==================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: this file is not needed at runtime but helps with code completion
    during development. Otherwise, pycharm shows numerous warning messages.
"""


# ============================================================
#  Imports
# ============================================================

import numpy as np
import jax
import jax.numpy as jnp
import os
import time

# Import configurations (namelists)
from ..config.Config import *
from ..config import Config

# Import derived variables
from ..config.DerivedVars import *
from ..config import DerivedVars

# Imports from initialization
from ..initialization.Initialization import Initialize_uvw, Initialize_TH
from ..initialization.Initialization import Initialize_GeoWind
from ..initialization.Initialization import Initialize_RayleighDampingLayer
from ..initialization.Preprocess import Wavenumber, Constant
from ..initialization.Preprocess import ZeRo3DIni, ZeRo2DIni, ZeRo1DIni
from ..initialization.Preprocess import ZeRo3D_padIni, ZeRo3D_fftIni, ZeRo3D_pad_fftIni

# Imports from operations
from ..operations.Derivatives import Derivxy, Derivz_M, velocityGradients
from ..operations.Derivatives import Derivz_TH, potentialTemperatureGradients
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
from ..subgridscale.SGSStresses import StressesUVPnodes_Dealias, StressesWnodes_Dealias
from ..subgridscale.SGSStresses import StressesUVPnodes_NoDealias, StressesWnodes_NoDealias
from ..subgridscale.SGSStresses import Wall
from ..subgridscale.DynamicSGS_Main import DynamicSGS, DynamicSGSscalar
from ..subgridscale.DynamicSGS_LASDD import LASDD
from ..subgridscale.DynamicSGS_ScalarLASDD import ScalarLASDD

# Imports from surface
from ..surface.SurfaceFlux import MOSTstable, MOSTunstable
from ..surface.SurfaceFlux import SurfaceFlux_HomogeneousConstantFlux
from ..surface.SurfaceFlux import SurfaceFlux_HeterogeneousConstantFlux

# Imports from pde
from ..pde.NSE_AdvectionTerms import Advection
from ..pde.SCL_AdvectionTerms import ScalarAdvection
from ..pde.NSE_BuoyancyTerms import BuoyancyOpt1, BuoyancyOpt2
from ..pde.NSE_PressureTerms import PressureInit, PressureRC
from ..pde.NSE_PressureTerms import PressureMatrix, PressureSolve
from ..pde.NSE_SGSTerms import DivStressStaticSGS, DivStressDynamicSGS
from ..pde.SCL_SGSTerms import DivFluxStaticSGS, DivFluxDynamicSGS
from ..pde.NSE_AllTerms import RHS_Momentum
from ..pde.SCL_AllTerms import RHS_Scalar
from ..pde.NSE_TimeAdvancement import AB2_uvw
from ..pde.SCL_TimeAdvancement import AB2_TH
