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
File: Main.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-5
:Description: main file for JAX-ALFA
"""


# ============================================================
#  Imports
# ============================================================
# This file is for run time
from .config.Imports import ImportLES
ImportLES()

# Import derived variables
from .config.DerivedVars import *

# This file is for IDE static analysis during development time
from .utilities.Pycharm import *

import shutil

# ============================================================
#  Initialize Static Variables
# ============================================================

kx2, ky2 = Wavenumber()
ZeRo3D = ZeRo3DIni()
ZeRo2D = ZeRo2DIni()
ZeRo1D = ZeRo1DIni()
ZeRo3D_fft = ZeRo3D_fftIni()
ZeRo3D_pad = ZeRo3D_padIni()
ZeRo3D_pad_fft = ZeRo3D_pad_fftIni()

# Static variables related to pressure solver
(kr2_pressure, kc2_pressure,
 a_pressure, b_pressure, c_pressure) = PressureInit()


# ============================================================
#  Initialize velocity, temperature, etc.
# ============================================================

u, v, w = Initialize_uvw()
TH = Initialize_TH()
Ug, Vg = Initialize_GeoWind()
RayleighDampCoeff, RayleighDampCoeff_stag = (
    Initialize_RayleighDampingLayer())

RHS_u_previous = ZeRo3D.copy()
RHS_v_previous = ZeRo3D.copy()
RHS_w_previous = ZeRo3D.copy()

RHS_TH_previous = ZeRo3D.copy()

CFLmax = 0
CFLmax_iteration = 1

# ============================================================
#  Initialize surface variables
# ============================================================
psi2D_m = ZeRo2D.copy()
psi2D_m0 = ZeRo2D.copy()
psi2D_h = ZeRo2D.copy()
psi2D_h0 = ZeRo2D.copy()
fi2D_m = 1.0 + ZeRo2D.copy()
fi2D_h = 1.0 + ZeRo2D.copy()

MOSTfunctions = (psi2D_m, psi2D_m0,
                 psi2D_h, psi2D_h0,
                 fi2D_m, fi2D_h)


# ============================================================
# Initialize statistics variables
# ============================================================
StatsDict = InitializeStats(nz, Ugal, ZeRo1D)
SampleCounter = 0  # Counter to sample statistics
# dumDir is identified based on the location of Main.py
dumDir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
OutputDir = os.path.join(dumDir, 'output')
# Delete all the pre-existing files in output directory
shutil.rmtree(OutputDir)
# Recreate the output directory
os.makedirs(OutputDir)


# ============================================================
#  Main simulation loop
# ============================================================

tic_tot = time.time()

for iteration in range(istep, nsteps+1, 1):

    if iteration > istep:

        RHS_u_previous = RHS_u
        RHS_v_previous = RHS_v
        RHS_w_previous = RHS_w

        RHS_TH_previous = RHS_TH

    # ------------------------------------------------------------
    #  Filtering and FFT Computations
    # ------------------------------------------------------------
    u, u_fft = Filtering_Explicit(FFT(u))
    v, v_fft = Filtering_Explicit(FFT(v))
    w, w_fft = Filtering_Explicit(FFT(w))

    TH, TH_fft = Filtering_Explicit(FFT(TH))

    # ------------------------------------------------------------
    #  Compute Surface Fluxes
    # ------------------------------------------------------------
    if optSurfFlux == 0:
        (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
            SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions))
    elif optSurfFlux == 1:
        (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
            SurfaceFlux_HeterogeneousConstantFlux(u, v, TH, MOSTfunctions))

    # ------------------------------------------------------------
    #  Compute Velocity Gradients
    # ------------------------------------------------------------
    (dudx, dvdx, dwdx,
     dudy, dvdy, dwdy,
     dudz, dvdz, dwdz) = (
        velocityGradients(
            u, v, w,
            u_fft, v_fft, w_fft,
            kx2, ky2,
            ustar, M_sfc_loc, MOSTfunctions,
            ZeRo3D))

    (dTHdx, dTHdy, dTHdz) = (
        potentialTemperatureGradients(
            TH, TH_fft,
            kx2, ky2,
            ustar, qz_sfc, MOSTfunctions,
            ZeRo3D))

    # ------------------------------------------------------------
    #  Compute Advection Terms
    # ------------------------------------------------------------
    Cx, Cy, Cz = Advection(
        u, v, w,
        dudy, dudz, dvdx, dvdz, dwdx, dwdy,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad,
        ZeRo3D_pad_fft)

    THAdvectionSum = ScalarAdvection(
        u, v, w,
        dTHdx, dTHdy, dTHdz,
        ZeRo3D, ZeRo3D_fft, ZeRo3D_pad,
        ZeRo3D_pad_fft)

    # ------------------------------------------------------------
    #  Compute Buoyancy Terms
    # ------------------------------------------------------------
    # FIXME: Humidity will be included in a later version
    H = ZeRo3D
    if optBuoyancy == 1:
        buoyancy = BuoyancyOpt1(TH, H, ZeRo3D)
    else:
        buoyancy = BuoyancyOpt2(TH, H, ZeRo3D)

    # ------------------------------------------------------------
    #  Compute SGS Terms
    # ------------------------------------------------------------

    if iteration == istep or iteration % dynamicSGS_call_time == 0:

        # print('Dynamic SGS')

        (divtx, divty, divtz,
         Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D,
         dynamicSGSmomentum) = (
            DivStressDynamicSGS(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                u, v, w, M_sfc_loc, MOSTfunctions,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        (qz, divq, Cs2PrRatio_1D, beta2_1D) = (
            DivFluxDynamicSGS(
                dynamicSGSmomentum[9:],
                TH,
                dTHdx, dTHdy, dTHdz,
                qz_sfc,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        # Unpack variables for computation of statistics
        _, _, _, txy, txz, tyz = dynamicSGSmomentum[0:6]

    else:

        # print('Static SGS')

        (divtx, divty, divtz,
         staticSGSmomentum) = (
            DivStressStaticSGS(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                Cs2_3D,
                u, v, M_sfc_loc, MOSTfunctions,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        qz, divq = (
            DivFluxStaticSGS(
                staticSGSmomentum[6:],
                Cs2PrRatio_3D,
                dTHdx, dTHdy, dTHdz,
                qz_sfc,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        # Unpack variables for computation of statistics
        _, _, _, txy, txz, tyz = staticSGSmomentum[0:6]

    # ------------------------------------------------------------
    #  Compute right hand side (RHS) terms
    # ------------------------------------------------------------

    (RHS_u, RHS_v, RHS_w) = (
        RHS_Momentum(u, v, w,
                     Ug, Vg,
                     Cx, Cy, Cz,
                     buoyancy,
                     divtx, divty, divtz,
                     RayleighDampCoeff, RayleighDampCoeff_stag))

    RHS_TH = RHS_Scalar(TH, THAdvectionSum, divq, RayleighDampCoeff_stag)

    # ------------------------------------------------------------
    #  Pressure solution
    # ------------------------------------------------------------

    (RC_real, RC_imag, fRz_real) = (
        PressureRC(
            u, v, w,
            RHS_u, RHS_v, RHS_w,
            RHS_u_previous, RHS_v_previous, RHS_w_previous,
            divtz, kr2_pressure, kc2_pressure))

    (p, dpdx, dpdy, dpdz) = (
        PressureSolve(
            RC_real, RC_imag, fRz_real,
            a_pressure, b_pressure, c_pressure))

    # Add pressure gradient terms to RHS
    RHS_u = RHS_u - dpdx
    RHS_v = RHS_v - dpdy
    RHS_w = RHS_w - dpdz

    # ------------------------------------------------------------
    #  Initialize RHS terms for previous time step
    # ------------------------------------------------------------

    if iteration == istep:
        RHS_u_previous = RHS_u
        RHS_v_previous = RHS_v
        RHS_w_previous = RHS_w

        RHS_TH_previous = RHS_TH

    # ------------------------------------------------------------
    #  Time advancement
    # ------------------------------------------------------------

    (u, v, w) = (
        AB2_uvw(u, v, w,
                RHS_u, RHS_u_previous,
                RHS_v, RHS_v_previous,
                RHS_w, RHS_w_previous))

    (TH) = (
        AB2_TH(TH,
               RHS_TH, RHS_TH_previous))

    # ------------------------------------------------------------
    #  Compute CFLmax
    # ------------------------------------------------------------
    CFLx = jnp.max(jnp.abs(u)) * dt_nondim / dx
    CFLy = jnp.max(jnp.abs(v)) * dt_nondim / dy
    CFLz = jnp.max(jnp.abs(w)) * dt_nondim / dz
    CFL = jnp.max(jnp.array([CFLx, CFLy, CFLz]))
    if CFL > CFLmax:
        CFLmax = CFL
        CFLmax_iteration = iteration

    # ------------------------------------------------------------
    #  Compute and output averaged statistics
    # ------------------------------------------------------------

    # Collect samples at specified intervals including output intervals
    if iteration % SampleInterval == 0:
        # Accumulation of statistics
        ResetFlag = 0
        StatsDict = ComputeStats(u, v, w, TH, txy, txz, tyz, qz,
                                 StatsDict, ResetFlag, ZeRo3D)
        SampleCounter += 1

        print(f"\n============= Finished Iteration {iteration} =============")
        print(
            f"Statistics: collected sample {SampleCounter} at iteration {iteration}")
        print(f"  Friction Velocity:    {jnp.mean(ustar):.2f} m/s")
        print(f"  Sensible Heat Flux:   {jnp.mean(qz_sfc):.2f} K m/s")
        print(f"  Current CFL:          {CFL:.3f}")
        print(f"  CFLmax:               {CFLmax:.3f}")
        print(f"  CFLmax happened at iteration: {CFLmax_iteration}")

    # At output intervals, check if we've collected any samples
    if iteration % OutputInterval == 0 and SampleCounter > 0:
        OutputStats = {}
        for key in StatsDict:
            if key not in ["Ugal", "ZeRo1D"]:
                # Average the accumulated statistics
                OutputStats[key] = StatsDict[key] / SampleCounter
            else:
                OutputStats[key] = StatsDict[key]

        # Generate output filename and save statistics
        OutputFile = f'ALFA_Statistics_Iteration_{iteration}.npz'
        OutputDirFile = os.path.join(OutputDir, OutputFile)
        np.savez(OutputDirFile, **OutputStats)
        print(
            f"Statistics saved to {OutputFile} "
            f"(averaged over {SampleCounter} samples)")

        # Reset statistics for next averaging interval
        SampleCounter = 0
        ResetFlag = 1
        StatsDict = ComputeStats(u, v, w, TH, txy, txz, tyz, qz,
                                 StatsDict, ResetFlag, ZeRo3D)

    # At regular intervals, save 3D fields for visualizations
    # Output 3D fields at specified intervals
    if iteration % Output3DInterval == 0:
        # Create dictionary of fields to save
        Fields3D = {
            "u": u + Ugal,
            # Add Galilean transformation velocity back to u component
            "v": v,
            "w": w,
            "TH": TH
        }

        # Generate output filename and save 3D fields
        OutputFile3D = f'ALFA_3DFields_Iteration_{iteration}.npz'
        OutputDirFile3D = os.path.join(OutputDir, OutputFile3D)
        np.savez(OutputDirFile3D, **Fields3D)
        print(f"3D fields saved to {OutputFile3D}")

print(f"Total Elapsed Time: {time.time() - tic_tot:.5f} seconds")
