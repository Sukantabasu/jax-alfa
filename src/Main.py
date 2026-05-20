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
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
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
if optMoisture >= 1:
    Q = Initialize_Q()
    RHS_Q_previous = ZeRo3D.copy()
    if optMoistureSurfBC >= 1:
        MoistureSurfaceBC_series = Initialize_MoistureSurfaceBC()
    Qadv = ZeRo3D
else:
    Q = ZeRo3D
    Qadv = ZeRo3D

if optGeoWind == 0:
    Ug, Vg = Initialize_GeoWind()
else:
    GeoWind_U, GeoWind_V = Initialize_GeoWind_Varying()
    Ug = jnp.broadcast_to(GeoWind_U[istep - 1].reshape(1, 1, nz), (nx, ny, nz))
    Vg = jnp.broadcast_to(GeoWind_V[istep - 1].reshape(1, 1, nz), (nx, ny, nz))

RayleighDampCoeff, RayleighDampCoeff_stag = (
    Initialize_RayleighDampingLayer())

RHS_u_previous  = ZeRo3D.copy()
RHS_v_previous  = ZeRo3D.copy()
RHS_w_previous  = ZeRo3D.copy()
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

# Load time-varying surface BC series once before the loop (optSurfBC >= 1)
if optSurfBC >= 1:
    SurfaceBC_series = Initialize_SurfaceBC()

# Load large-scale advection forcing once before the loop (optAdvection >= 1)
if optAdvection >= 1:
    AdvForcing_U, AdvForcing_V, AdvForcing_TH, AdvForcing_Q = Initialize_AdvForcing()
    Uadv  = jnp.broadcast_to(AdvForcing_U[istep - 1].reshape(1, 1, nz), (nx, ny, nz))
    Vadv  = jnp.broadcast_to(AdvForcing_V[istep - 1].reshape(1, 1, nz), (nx, ny, nz))
    THadv = jnp.broadcast_to(AdvForcing_TH[istep - 1].reshape(1, 1, nz), (nx, ny, nz))
    if optMoisture >= 1:
        Qadv  = jnp.broadcast_to(AdvForcing_Q[istep - 1].reshape(1, 1, nz), (nx, ny, nz))
else:
    Uadv  = ZeRo3D
    Vadv  = ZeRo3D
    THadv = ZeRo3D


# ============================================================
# Initialize statistics variables
# ============================================================
StatsDict = InitializeStats(ZeRo1D)
SampleCounter = 0  # Counter to sample statistics
OutputDir = os.path.join(os.environ['JAXALFA_RUNDIR'], 'output')
os.makedirs(OutputDir, exist_ok=True)


# ============================================================
#  Main simulation loop
# ============================================================

tic_tot = time.time()

for iteration in range(istep, nsteps+1, 1):

    if iteration > istep:

        RHS_u_previous  = RHS_u
        RHS_v_previous  = RHS_v
        RHS_w_previous  = RHS_w
        RHS_TH_previous = RHS_TH
        if optMoisture >= 1:
            RHS_Q_previous = RHS_Q

    # ------------------------------------------------------------
    #  Update time/height-varying geostrophic wind (optGeoWind >= 1)
    # ------------------------------------------------------------
    if optGeoWind >= 1:
        Ug = jnp.broadcast_to(
            GeoWind_U[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))
        Vg = jnp.broadcast_to(
            GeoWind_V[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))

    # ------------------------------------------------------------
    #  Update time/height-varying large-scale advection (optAdvection >= 1)
    # ------------------------------------------------------------
    if optAdvection >= 1:
        Uadv  = jnp.broadcast_to(
            AdvForcing_U[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))
        Vadv  = jnp.broadcast_to(
            AdvForcing_V[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))
        THadv = jnp.broadcast_to(
            AdvForcing_TH[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))
        if optMoisture >= 1:
            Qadv = jnp.broadcast_to(
                AdvForcing_Q[iteration - 1].reshape(1, 1, nz), (nx, ny, nz))

    # ------------------------------------------------------------
    #  Filtering and FFT Computations
    # ------------------------------------------------------------
    u, u_fft = Filtering_Explicit(FFT(u))
    v, v_fft = Filtering_Explicit(FFT(v))
    w, w_fft = Filtering_Explicit(FFT(w))

    TH, _ = Filtering_Explicit(FFT(TH))
    if optMoisture >= 1:
        Q, _ = Filtering_Explicit(FFT(Q))

    # ------------------------------------------------------------
    #  Compute Surface Fluxes
    #
    #  All branches set:
    #    M_sfc_loc   (nx, ny)  surface wind speed
    #    ustar       (nx, ny)  friction velocity
    #    qz_sfc_step (nx, ny)  surface heat flux field  (qz = -u* x th*)
    #    qz_sfc_avg  scalar    planar-mean, non-dimensional
    #    invOB       (nx, ny)  inverse Obukhov length
    #    MOSTfunctions         updated stability functions
    # ------------------------------------------------------------

    if optSurfBC == 0:
        # Constant prescribed heat flux
        if optSurfFlux == 0:
            (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions))
        else:
            (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HeterogeneousConstantFlux(u, v, TH, MOSTfunctions))
        qz_sfc_step = qz_sfc  # global (nx,ny) array from DerivedVars

    elif optSurfBC == 1:
        # Time-varying prescribed heat flux
        sfc_val = SurfaceBC_series[iteration - 1]
        if optSurfFlux == 0:
            (M_sfc_loc, ustar, qz_sfc_step, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HomogeneousVaryingFlux(u, v, TH, sfc_val, MOSTfunctions))
        else:
            (M_sfc_loc, ustar, qz_sfc_step, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HeterogeneousVaryingFlux(u, v, TH, sfc_val, MOSTfunctions))

    else:
        # optSurfBC == 2: time-varying prescribed surface temperature
        sfc_val = SurfaceBC_series[iteration - 1]
        if optSurfFlux == 0:
            (M_sfc_loc, ustar, qz_sfc_step, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HomogeneousPrescribedTemperature(
                    u, v, TH, sfc_val, MOSTfunctions))
        else:
            (M_sfc_loc, ustar, qz_sfc_step, qz_sfc_avg, invOB, MOSTfunctions) = (
                SurfaceFlux_HeterogeneousPrescribedTemperature(
                    u, v, TH, sfc_val, MOSTfunctions))

    # ------------------------------------------------------------
    #  Compute Surface Moisture Flux (optMoisture >= 1)
    #
    #  Uses ustar and MOSTfunctions already computed above.
    #  All branches set:
    #    qm_sfc_step (nx, ny)   surface moisture flux field
    #    qm_sfc_avg  scalar     planar-mean, non-dimensional
    # ------------------------------------------------------------
    if optMoisture >= 1:
        if optMoistureSurfBC == 0:
            qm_sfc_step = qm_sfc          # constant from DerivedVars
        elif optMoistureSurfBC == 1:
            qm_sfc_t    = MoistureSurfaceBC_series[iteration - 1]
            qm_sfc_step = qm_sfc_t * jnp.ones((nx, ny))
        else:  # optMoistureSurfBC == 2: prescribed surface Q
            Q_sfc_t = MoistureSurfaceBC_series[iteration - 1]
            if optSurfFlux == 0:
                qm_sfc_step = SurfaceMoistureFlux_HomogeneousPrescribedQ(
                    Q, ustar, Q_sfc_t, MOSTfunctions)
            else:
                qm_sfc_step = SurfaceMoistureFlux_HeterogeneousPrescribedQ(
                    Q, ustar, Q_sfc_t, MOSTfunctions)
        qm_sfc_avg = jnp.mean(qm_sfc_step)
    else:
        qm_sfc_step = ZeRo2D
        qm_sfc_avg  = 0.0

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
            TH,
            kx2, ky2,
            ustar, qz_sfc_step, MOSTfunctions,
            ZeRo3D))

    if optMoisture >= 1:
        (dQdx, dQdy, dQdz) = moistureGradients(
            Q, kx2, ky2, ustar, qm_sfc_step, MOSTfunctions, ZeRo3D)
    else:
        dQdx = ZeRo3D; dQdy = ZeRo3D; dQdz = ZeRo3D

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

    if optMoisture >= 1:
        QAdvectionSum = ScalarAdvection(
            u, v, w,
            dQdx, dQdy, dQdz,
            ZeRo3D, ZeRo3D_fft, ZeRo3D_pad,
            ZeRo3D_pad_fft)

    # ------------------------------------------------------------
    #  Compute Buoyancy Terms
    # ------------------------------------------------------------
    H = Q if optMoisture >= 1 else ZeRo3D
    if optBuoyancy == 0:
        buoyancy = BuoyancyOpt1(TH, H, ZeRo3D)
    else:
        buoyancy = BuoyancyOpt2(TH, H, ZeRo3D)

    # ------------------------------------------------------------
    #  Compute SGS Terms
    # ------------------------------------------------------------

    if optSgs >= 1 and (iteration == istep or iteration % dynamicSGS_call_time == 0):

        # print('Dynamic SGS')

        (divtx, divty, divtz,
         Cs2_1D_avg1, Cs2_1D_avg2, beta1_1D,
         Cs2_3D,
         dynamicSGSmomentum) = (
            DivStressDynamicSGS(
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                u, v, w, M_sfc_loc, MOSTfunctions,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        (qz, divq, Cs2PrRatio_3D, Cs2PrRatio_1D, beta2_1D) = (
            DivFluxDynamicSGS(
                dynamicSGSmomentum[10:],
                TH,
                dTHdx, dTHdy, dTHdz,
                qz_sfc_step,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        # Moisture SGS: reuse strain rates from dynamic momentum SGS with
        # the same Cs2PrRatio (turbulent Sc = turbulent Pr approximation).
        # dynamicSGSmomentum[19:23] = (S_uvp, S_uvp_pad, S_w, S_w_pad)
        if optMoisture >= 1:
            qHz_q, divqm = DivFluxStaticSGS(
                (dynamicSGSmomentum[19], dynamicSGSmomentum[20],
                 dynamicSGSmomentum[21], dynamicSGSmomentum[22]),
                Cs2PrRatio_3D,
                dQdx, dQdy, dQdz,
                qm_sfc_step,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2)
        else:
            qHz_q = ZeRo3D; divqm = ZeRo3D

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
                qz_sfc_step,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2))

        # Moisture SGS: same Cs2PrRatio as heat.
        if optMoisture >= 1:
            qHz_q, divqm = DivFluxStaticSGS(
                staticSGSmomentum[6:],
                Cs2PrRatio_3D,
                dQdx, dQdy, dQdz,
                qm_sfc_step,
                ZeRo3D, ZeRo3D_fft, ZeRo3D_pad_fft,
                kx2, ky2)
        else:
            qHz_q = ZeRo3D; divqm = ZeRo3D

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
                     RayleighDampCoeff, RayleighDampCoeff_stag,
                     Uadv, Vadv))

    RHS_TH = RHS_Scalar(TH, THAdvectionSum, divq, RayleighDampCoeff_stag, THadv)
    if optMoisture >= 1:
        RHS_Q = RHS_Moisture(Q, QAdvectionSum, divqm, RayleighDampCoeff_stag, Qadv)

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
        RHS_u_previous  = RHS_u
        RHS_v_previous  = RHS_v
        RHS_w_previous  = RHS_w
        RHS_TH_previous = RHS_TH
        if optMoisture >= 1:
            RHS_Q_previous = RHS_Q

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

    if optMoisture >= 1:
        Q = AB2_Q(Q, RHS_Q, RHS_Q_previous)

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
        StatsDict = ComputeStats(u, v, w, TH, Q,
                                 dudz, dvdz, dTHdz, dQdz,
                                 M_sfc_loc, ustar, qz_sfc_avg, qm_sfc_avg,
                                 txy, txz, tyz, qz, qHz_q,
                                 Cs2_1D_avg1, Cs2_1D_avg2,
                                 Cs2PrRatio_1D,
                                 beta1_1D, beta2_1D,
                                 StatsDict, ResetFlag,
                                 ZeRo3D)
        SampleCounter += 1

        pct     = 100.0 * iteration / nsteps
        elapsed = time.time() - tic_tot
        rate    = elapsed / (iteration - istep + 1)   # seconds per iteration
        eta     = rate * (nsteps - iteration)

        def _fmt(s):
            h, r = divmod(int(s), 3600)
            m, sec = divmod(r, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        print(f"\n============= Finished Iteration {iteration} / {nsteps} "
              f"({pct:.1f}%) =============")
        print(f"  Elapsed: {_fmt(elapsed)}   ETA: {_fmt(eta)}")
        print(
            f"Statistics: collected sample {SampleCounter} at iteration {iteration}")
        print(f"  Friction Velocity:    {jnp.sqrt(jnp.mean(ustar ** 2)):.4f} "
              f"m/s")
        print(f"  Sensible Heat Flux:   "
              f"{float(qz_sfc_avg * u_scale * TH_scale):.4f} K m/s")
        if optMoisture >= 1:
            print(f"  Moisture Flux:        "
                  f"{float(qm_sfc_avg * u_scale * Q_scale):.6e} kg/kg m/s")
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
        StatsDict = ComputeStats(u, v, w, TH, Q,
                                 dudz, dvdz, dTHdz, dQdz,
                                 M_sfc_loc, ustar, qz_sfc_avg, qm_sfc_avg,
                                 txy, txz, tyz, qz, qHz_q,
                                 Cs2_1D_avg1, Cs2_1D_avg2,
                                 Cs2PrRatio_1D,
                                 beta1_1D, beta2_1D,
                                 StatsDict, ResetFlag,
                                 ZeRo3D)

    # At regular intervals, save 3D fields for visualizations
    # Output 3D fields at specified intervals
    if iteration % Output3DInterval == 0:
        # Create dictionary of fields to save
        Fields3D = {
            "u": u + Ugal,        # Galilean velocity added back
            "v": v,
            "w": w,
            "TH": TH + T_0_nondim  # anomaly → absolute (TH stored as TH - T_0)
        }
        if optMoisture >= 1:
            Fields3D["Q"] = Q

        # Generate output filename and save 3D fields
        OutputFile3D = f'ALFA_3DFields_Iteration_{iteration}.npz'
        OutputDirFile3D = os.path.join(OutputDir, OutputFile3D)
        np.savez(OutputDirFile3D, **Fields3D)
        print(f"3D fields saved to {OutputFile3D}")

print(f"Total Elapsed Time: {time.time() - tic_tot:.5f} seconds")
