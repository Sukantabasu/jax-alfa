# JAX-ALFA Code Review Report

**Prepared by:** Claude (Anthropic)  
**Date:** April 30, 2026  
**Version reviewed:** JAXALFA 0.1  

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [Numerical Methods — In Depth](#3-numerical-methods--in-depth)
4. [Subgrid-Scale Modeling — In Depth](#4-subgrid-scale-modeling--in-depth)
5. [Surface Layer — Current Implementation](#5-surface-layer--current-implementation)
6. [Potential Bugs and Code Improvements](#6-potential-bugs-and-code-improvements)
7. [Surface Layer Extension: Time-Varying Temperature and Fluxes](#7-surface-layer-extension-time-varying-temperature-and-fluxes)
8. [Running the Code: Workflow and I/O Management](#8-running-the-code-workflow-and-io-management)
9. [Suggested Input/Output Folder Strategy](#9-suggested-inputoutput-folder-strategy)

---

## 1. Framework Overview

JAX-ALFA is a pseudo-spectral Large-Eddy Simulation (LES) code for the atmospheric boundary layer (ABL). It solves the filtered incompressible Navier-Stokes equations under the Boussinesq approximation. Its distinguishing feature is its use of **JAX** as the sole computational backend, enabling JIT compilation, automatic differentiation, and transparent CPU/GPU execution from the same source code.

### Design Philosophy

The code is built around three JAX principles:

- **Pure functional style.** Nearly every computation is written as a function that receives its inputs as arguments and returns its outputs. There is almost no mutable global state inside JIT-compiled functions.
- **Pre-allocated arrays.** Because JAX's JIT compilation traces array shapes at compile time, zero arrays (`ZeRo3D`, `ZeRo3D_fft`, `ZeRo3D_pad`, `ZeRo3D_pad_fft`) are created once at startup and passed through the call graph as template buffers for `.copy()` operations.
- **Functional import injection.** `Imports.py::ImportLES()` uses Python's `inspect` module to reach into its caller's global namespace and inject all symbols there. This unusual pattern means `Main.py` carries no explicit `import` statements yet has full access to every function in the codebase.

### Governing Equations

The code solves the filtered Boussinesq Navier-Stokes equations:

```
∂u_i/∂t + u_j ∂u_i/∂x_j = -∂p/∂x_i + B δ_{i3} - f ε_{ij3} (u_j - U_g,j) - ∂τ_{ij}/∂x_j - D_i

∂θ/∂t + u_j ∂θ/∂x_j = -∂q_j/∂x_j - D_θ

∂u_i/∂x_i = 0
```

Where:
- `B = g (θ_v − ⟨θ_v⟩) / ⟨θ_v⟩` (buoyancy, option 2) or `g (θ_v − ⟨θ_v⟩) / θ_0` (option 1)
- `τ_{ij}` = SGS momentum stress, `q_j` = SGS scalar flux
- `D_i`, `D_θ` = Rayleigh damping terms
- `f` = Coriolis parameter
- `U_g` = geostrophic wind

The equations are non-dimensionalized using the length scale `z* = L_x / (2π)` and velocity scale `u* = 1` m/s, so that horizontal wavenumbers are integers.

---

## 2. Architecture and Data Flow

### Module Dependency Graph

```
ConfigLoader.py  ──►  DerivedVars.py  ──►  (all other modules)
                Imports.py       ──►  Main.py  (runtime namespace)
                Utilities.py     ──►  (statistics, LASDD)
                Preprocess.py    ──►  (FFT, Derivatives, Dealiasing)
                
Main.py calls (per timestep):
  Filtering_Explicit  →  FFT results
  SurfaceFlux_*       →  (ustar, M_sfc, MOSTfunctions)
  velocityGradients   →  (all 9 velocity gradient components)
  potTemperatureGrad  →  (dTHdx, dTHdy, dTHdz)
  Advection           →  (Cx, Cy, Cz)
  ScalarAdvection     →  THAdvectionSum
  BuoyancyOpt*        →  buoyancy
  DivStressDynamic/StaticSGS  →  (divtx, divty, divtz) + SGS intermediates
  DivFluxDynamic/StaticSGS    →  (qz, divq)
  RHS_Momentum        →  (RHS_u, RHS_v, RHS_w)
  RHS_Scalar          →  RHS_TH
  PressureRC + PressureSolve  →  (p, dpdx, dpdy, dpdz)
  AB2_uvw, AB2_TH     →  (u_new, v_new, w_new, TH_new)
  ComputeStats        →  StatsDict
```

### Staggered Grid Layout

The vertical grid uses a standard Arakawa C-staggering:

```
Level index k:    0        1        2      ...    nz-2    nz-1
                  
Full levels (w):  z=0      z=dz     z=2dz          z=(nz-2)dz   z=(nz-1)dz
                  |        |        |               |             |
Half levels (u,v,θ): z=0.5dz   z=1.5dz            z=(nz-1.5)dz z=(nz-0.5)dz
```

- `u`, `v`, `θ` are stored at **half levels**: `z_k = (k + 0.5) dz`  
- `w` is stored at **full levels**: `z_k = k dz`
- Wall: `w[k=0] = 0`  
- Top: `w[k=nz-1] = 0`

`StagGridAvg(F)` returns `0.5 * (F[:,:,:-1] + F[:,:,1:])`, which is the average of adjacent half-levels evaluated at a full level, or vice versa.

### Galilean Transformation

At startup, `u` is loaded as `u_file - Ugal`. The simulation evolves in the moving frame. `Ugal` is added back only for:
- 3D field output (`"u": u + Ugal` in the output dictionary)
- Statistics (the `mU` profile adds `Ugal`)
- The Coriolis term: `f * (Ug - Ugal - u)` in `NSE_AllTerms.py`

This reduces the CFL number when the mean flow is near `Ugal`.

### Non-Dimensionalization

| Quantity | Scale | Code symbol |
|---------|-------|-------------|
| Length | `z* = L_x / (2π)` | `z_scale` |
| Velocity | `u* = 1` m/s | `u_scale` |
| Temperature | `θ* = 1` K | `TH_scale` |
| Time | `t* = z* / u*` | (implicit) |
| Wavenumbers | Integers (0..nx/2) | `kx2`, `ky2` |

The horizontal wavenumbers are pure integers because `dx = 2π/nx` in the non-dimensional system. This is why `Derivxy` multiplies by `1j * kxy2` without any additional scaling.

---

## 3. Numerical Methods — In Depth

### 3.1 Spectral Derivatives (x, y)

Horizontal derivatives are computed exactly by:

```
∂F/∂x ≡ IFFT(i kx · F̂)
```

The Nyquist frequency component is explicitly zeroed in `Wavenumber()` to prevent aliasing from odd-order derivatives. The `rfft2` is used along axes `(0,1)` (x and y), leaving the z-axis in physical space. This gives a "2D spectral, 1D physical" representation throughout.

### 3.2 Vertical Finite Differences

Second-order centered differences are used in the vertical. Because `u` and `w` live on different grids, care is taken:

- `dudz` at full level k = `(u[k] - u[k-1]) / dz` — centered since `u` is at half-levels
- `dwdz` at half level k = `(w[k+1] - w[k]) / dz` — centered since `w` is at full levels

Bottom boundary conditions use the MOST wall law rather than explicit BCs.

### 3.3 Dealiasing (3/2 Rule)

When `FGR = 1`, aliasing is removed by zero-padding in Fourier space before multiplying fields:

**`Dealias1(F_fft, ...)` — expand to padded grid:**
- Inserts `F_fft` into a zero array of size `(3nx/2, 3ny/2)` by copying the positive and negative frequency quadrants.
- Applies `irfft2` → physical field on the 3/2-size grid.

**`Dealias2(F_pad_fft, ...)` — contract back:**
- Extracts low-frequency quadrants from the padded FFT.
- Applies `irfft2` with scale factor `9/4` (compensates for the 3/2 factor from each direction: `(3/2)² = 9/4`).

For `FGR ≥ 2`, dealiasing is disabled and a wider explicit filter is applied instead.

### 3.4 Filtering Hierarchy

Three filter levels are defined, all as spectral sharp-cutoff filters:

| Function | Cutoff (x) | Cutoff (y) | Purpose |
|----------|-----------|-----------|---------|
| `Filtering_Explicit` | `nx/(2·FGR)` | `ny/(2·FGR)` | Removes Nyquist every step |
| `Filtering_Level1` | `nx/(2·FGR·TFR)` | `ny/(2·FGR·TFR)` | Test filter for LASDD (width = FGR·TFR) |
| `Filtering_Level2` | `nx/(2·FGR·TFR²)` | `ny/(2·FGR·TFR²)` | Double test filter for LASDD (width = FGR·TFR²) |

When `FGR = 1`, `TFR = 2`, so Level1 cuts at `nx/4` and Level2 at `nx/8`.  
When `FGR ≥ 2`, `TFR = √2`.

### 3.5 Advection (Curl Form)

The nonlinear advection is written in the rotation (curl) form to conserve discrete kinetic energy better than the divergence or convective forms:

```
Cx = v (∂u/∂y − ∂v/∂x) + w (∂u/∂z − ∂w/∂x)
Cy = u (∂v/∂x − ∂u/∂y) + w (∂v/∂z − ∂w/∂y)
Cz = u (∂w/∂x − ∂u/∂z) + v (∂w/∂y − ∂v/∂z)
```

The z-component `Cz` requires interpolation of `u` and `v` from half levels to the full-level w-grid using `StagGridAvg`. The wall (Cz=0) and top (Cz=0) boundary conditions on the w-equation are enforced explicitly.

### 3.6 Pressure Solver

The pressure Poisson equation is solved spectrally in x and y, and with a tridiagonal Thomas algorithm in z. The implementation is notable for its batching strategy inside `PressureSolve`:

1. All `(nx × ny_rfft)` wavenumber pairs `(i, j)` each require solving an `(nz+1) × (nz+1)` tridiagonal system.
2. These are batched into groups of 8 (parameter `batch_size`) processed with nested `jax.vmap` calls, then iterated with `jax.lax.scan`.
3. Special modes — `(0,0)` (zero mode), Nyquist in x, and Nyquist in y — are handled by `jax.lax.cond` returning zero (for Nyquist) or a cumulative-sum integration of `fRz` (for the zero mode pressure constant).

The zero mode `(0,0)` is physically the horizontally-averaged pressure, which is undetermined up to a constant. The code sets it via the first-level value and cumulative integration of `fRz`, fixing the gauge.

### 3.7 Time Integration

Adams-Bashforth 2nd-order (AB2):

```
u^(n+1) = u^n + Δt [1.5 RHS^n − 0.5 RHS^(n−1)]
```

On the first iteration (`istep`), `RHS_previous = RHS` is used, reducing to forward Euler. After pressure correction, the final form is:

```
RHS_intermediate = RHS + (2/3Δt) u − (1/3) RHS_previous
```

used in `PressureRC` to compute a consistent discrete divergence for the Poisson equation.

---

## 4. Subgrid-Scale Modeling — In Depth

Three SGS options are available. The default is LASDD.

### 4.1 Static Smagorinsky

```
τ_{ij} = −2 Cs² Δ² |S| S_{ij}
q_i = −2 (Cs²/Pr_t) Δ² |S| ∂θ/∂x_i
```

Fixed `Cs = 0.1`, `Pr_t = 1.0` (from Config.py). No dynamic adjustment.

### 4.2 Dynamic Smagorinsky (LASDD)

The **Locally-Averaged Scale-Dependent Dynamic (LASDD)** model, due to Bou-Zeid et al. (2005) and Porté-Agel et al. (2000), is the most sophisticated option. It:

1. Accounts for scale dependence of the Smagorinsky coefficient (the coefficient at the test-filter scale differs from that at the grid-filter scale by a factor `β^2`).
2. Uses local spatial averaging (3×3 box filter `Imfilter`) instead of horizontal planar averaging, preserving spatial variability of `Cs²`.

**Algorithm for momentum (LASDD function):**

**Step 1 — Interpolate w to half levels:**  
`w_` is interpolated from full to half levels using `StagGridAvg`.

**Step 2 — Construct filtered velocity products at two test levels:**  
Apply `Filtering_Level1` (width `Δ̂ = TFR·Δ`) and `Filtering_Level2` (width `Δ̂d = TFR²·Δ`) to all velocity components and their products `uu, vv, ww, uv, uw, vw`.

**Step 3 — Construct Leonard stress tensors:**  
```
L_{ij} = û_i û_j hat − û_i hat û_j hat   (at test level 1)
Q_{ij} = ũ_i ũ_j hat − ũ_i hatd ũ_j hatd  (at test level 2)
```
These represent the resolved subgrid stress at each test-filter level.

**Step 4 — Construct M tensors and find β₁:**  
The scale-dependent parameter `β₁` satisfies:
```
Cs²(Δ̂) = β₁² Cs²(Δ)
```
This leads to the contraction:
```
L_{ij} M_{ij}^β = Cs²(Δ) M_{ij}^β M_{ij}^β
```
where `M_{ij}^β = 2Δ² (|Ŝ|S_{ij}^hat) − 2β₁(TFR·Δ)² Ŝ·Ŝ_{ij}^hat`.

The self-consistency requirement at both test-filter levels produces a **5th-degree polynomial** in `β₁` with planar-averaged coefficients. This polynomial is solved per vertical level by **Laguerre's method** with 8 initial guesses in the range [0.5, 4.5].

**Step 5 — Compute Cs² locally:**  
```
Cs²(i,j,k) = LM_filtered(i,j,k) / MM_filtered(i,j,k)
```
where the numerator and denominator are smoothed by the 3×3 box filter. Values below 0 or above 1 are clipped to 0.

**Step 6 — Apply stresses:**  
```
τ_{ij} = −2 Cs²(i,j,k) Δ² |S| S_{ij}
```

The scalar LASDD in `DynamicSGS_ScalarLASDD.py` follows the identical pattern to compute `Cs²/Pr_t` (the scalar diffusivity coefficient), reusing the filtered velocity fields from the momentum LASDD to avoid redundant FFTs.

### 4.3 Mixed-Mode Operation

When `dynamicSGS_call_time > 1`, the dynamic LASDD runs on iteration `istep` and then every `dynamicSGS_call_time` steps; static Smagorinsky (with the configured constant `Cs²`) fills the intermediate steps. The dynamic SGS coefficients (`Cs2_1D_avg1`, `beta1_1D`, etc.) reported in statistics on static steps are the values from the most recent dynamic computation.

### 4.4 Wall Model

At the bottom boundary, SGS stresses `τ_{xz}` and `τ_{yz}` on the w-grid are not computed from the strain rate but from the wall model:

```
τ_{xz}|_wall = −u*² (u + Ugal) / M
τ_{yz}|_wall = −u*² v / M
```

where `u*` is the friction velocity from MOST and `M` is the near-surface wind speed. This implements a log-law boundary condition consistent with the surface flux computation.

---

## 5. Surface Layer — Current Implementation

### 5.1 Monin-Obukhov Similarity Theory

Surface fluxes are computed using MOST. The key relationship is the log + stability-correction wind profile:

```
M = (u*/κ) [ln(z₁/z₀m) + ψ_m(z₁/L) − ψ_m(z₀m/L)]
```

Inverted for `u*`:
```
u*(i,j) = κ M(i,j) / [ln(z₁/z₀m) + ψ_m(z₁/L) − ψ_m(z₀m/L)]
```

The Obukhov length `L` is computed from:
```
1/L = −κ g ⟨wθ⟩_sfc / (u*³ θ_sfc)
```

The stability functions currently implemented are:
- **Stable** (`qz_sfc ≤ 0`): Businger-Dyer linear forms `ψ_m = ψ_h = 5 z/L`
- **Unstable** (`qz_sfc > 0`): Businger-Dyer-Pandolfo forms

The stability flag uses a **single scalar condition** `is_stable = qz_sfc_avg <= 0`, which routes the entire domain through one branch via `jax.lax.cond`. This is efficient for homogeneous surfaces.

### 5.2 Homogeneous vs. Heterogeneous

Two surface flux functions exist:

| Function | Wind speed used | Temperature used |
|---------|----------------|-----------------|
| `SurfaceFlux_HomogeneousConstantFlux` | Planar average of `M` | Planar average of `θ_sfc` |
| `SurfaceFlux_HeterogeneousConstantFlux` | Local `M(i,j)` | Local `θ_sfc(i,j)` |

Both use the **same scalar stability branch** (`is_stable` based on mean flux), which is physically consistent with a prescribed-flux setup where the sign of the heat flux does not vary spatially.

### 5.3 Current Limitations

1. **`qz_sfc` is a compile-time constant.** It is defined in `DerivedVars.py` as:
   ```python
   qz_sfc = SensibleHeatFlux * jnp.ones((nx, ny)) / (u_scale * TH_scale)
   ```
   This value is set once at module import and cannot change during a simulation.

2. **No surface temperature boundary condition.** There is no option for a prescribed surface temperature `θ_s(t)` with bulk-formula heat flux. The GABLS and diurnal-cycle cases that require a cooling or warming surface temperature are not yet supported by `SurfaceFlux.py`.

3. **No radiation model.** The surface energy balance has no radiative forcing; the entire surface flux is prescribed directly.

4. **The Obukhov length iteration is not converged.** The code uses a single-pass update: it computes `invOB` from the current `u*`, then updates the stability functions, then computes a new `u*` — but does **not** iterate to convergence. In practice, with small timesteps, this lagged-update is adequate, but it is formally first-order accurate in time for the flux–stability coupling.

---

## 6. Potential Bugs and Code Improvements

### 6.1 Bug: Division by Zero in Surface Flux Near Startup

**File:** `src/surface/SurfaceFlux.py`  
**Lines:** `invOB` computation in both `SurfaceFlux_Homogeneous*` and `SurfaceFlux_Heterogeneous*`

```python
invOB = -(vonk * g_nondim * qz_sfc_avg) / ((ustar ** 3) * TH_sfc_loc)
```

During the first few timesteps, when the velocity field is initialized from a file, `ustar` can be very close to zero if `M` near the surface is small. Division by `ustar³` will produce infinity or NaN, which then propagates through the stability functions and into the velocity gradients. This could cause the simulation to blow up silently on timestep 1 with certain initial conditions.

**Recommended fix:** Guard `ustar` with a small positive floor before computing `invOB`:
```python
ustar_safe = jnp.maximum(ustar, 1e-6)
invOB = -(vonk * g_nondim * qz_sfc_avg) / ((ustar_safe ** 3) * TH_sfc_loc)
```
The same protection is advisable for `M_sfc_loc` in the denominator of the `Wall` function in `SGSStresses.py`.

---

### 6.2 Bug: `MOSTunstable` Missing `@jax.jit` Decorator

**File:** `src/surface/SurfaceFlux.py`, line 78

`MOSTstable` is decorated with `@jax.jit` but `MOSTunstable` is not. Since both are called exclusively inside already-JIT-compiled functions (`SurfaceFlux_HomogeneousConstantFlux` and `SurfaceFlux_HeterogeneousConstantFlux`), the missing decorator does not prevent correct execution. However, the inconsistency is confusing and could cause unexpected tracing overhead if `MOSTunstable` were ever called directly from outside a JIT context.

**Recommended fix:** Add `@jax.jit` to `MOSTunstable`.

---

### 6.3 Inefficiency: Redundant `Wavenumber()` Call Inside `PressureSolve`

**File:** `src/pde/NSE_PressureTerms.py`, line 326

```python
kx2, ky2 = Wavenumber()
```

`PressureSolve` is called every timestep from `Main.py`. Inside it, `Wavenumber()` recomputes and broadcasts the full `(nx, ny_rfft, nz)` wavenumber arrays. This is wasteful since `kx2` and `ky2` are already computed once in `Main.py`. The arrays should be passed in as arguments or computed once inside `PressureInit` and cached.

---

### 6.4 Inefficiency: Redundant FFT Array Allocation in Filtering Functions

**File:** `src/operations/Filtering.py`

All three filtering functions (`Filtering_Explicit`, `Filtering_Level1`, `Filtering_Level2`) call `ZeRo3D_fftIni()` internally:

```python
F_fft_new = ZeRo3D_fftIni()
```

This allocates a new `(nx, ny_rfft, nz)` complex array on every call. Since these functions run every timestep and are JIT-compiled, JAX should cache the allocation. However, the pre-allocated `ZeRo3D_fft` array is already available in the caller's namespace and could be passed in instead — consistent with how all other functions handle pre-allocation.

---

### 6.5 Inefficiency: Python For-Loops in `PressureInit`

**File:** `src/pde/NSE_PressureTerms.py`, lines 74–82

```python
for i in range(nx):
    if i < nx // 2:
        kr2_pressure = kr2_pressure.at[i, :].set(i)
    else:
        kr2_pressure = kr2_pressure.at[i, :].set(i - nx)
for j in range(ny_rfft):
    kc2_pressure = kc2_pressure.at[:, j].set(j)
```

`PressureInit` is decorated with `@jax.jit`, so these loops are unrolled at trace time. With `nx = 128` this creates 128 separate `.at[].set()` operations in the traced graph, which is verbose but harmless. The same result can be achieved cleanly with array operations, which would also be more readable.

---

### 6.6 Potential Issue: `PressureSolve` Nyquist Mode Handling with `vmap`

**File:** `src/pde/NSE_PressureTerms.py`, lines 246–261

Inside `solve_system(i, j, ...)`:
```python
is_zero_mode = (i == 0) & (j == 0)
is_nyquist_x = (i == nx_rfft - 1)
is_nyquist_y = (j == ny_rfft - 1)
is_special = is_zero_mode | is_nyquist_x | is_nyquist_y
```

When this function is called inside `jax.vmap`, `i` and `j` are JAX traced integers (0-dimensional arrays). Boolean operations on traced values produce traced booleans, and `jax.lax.cond` with a traced boolean condition is valid in JAX. However, `is_nyquist_x = (i == nx_rfft - 1)` compares a traced value against a Python integer — this is correct. The logic appears sound, but it is worth noting that for non-square domains (`nx ≠ ny`) or odd `nx`, the Nyquist detection must match exactly what `rfft2` produces. The code assumes `nx` is even, which is always satisfied by the standard powers-of-2 grid sizes, but is not explicitly validated.

---

### 6.7 Design Issue: Heterogeneous Surface Flux with Scalar Stability Branch

**File:** `src/surface/SurfaceFlux.py`, `SurfaceFlux_HeterogeneousConstantFlux`

```python
is_stable = qz_sfc_avg <= 0
psi2D_m, ... = jax.lax.cond(is_stable, ...)
```

For the heterogeneous case, each grid column (i,j) has its own `ustar(i,j)` and `invOB(i,j)`, but the stability branch (stable vs. unstable) is determined by the **domain-averaged** flux sign. If `qz_sfc` has spatially varying sign (e.g., mixed vegetated/bare-soil surface in future extensions), some columns should use the stable function while others use the unstable function. The current scalar `lax.cond` cannot handle this.

For the current use cases (spatially uniform `SensibleHeatFlux` from `Config.py`), `qz_sfc(i,j)` is constant across the domain, so this is not actually a bug. But it will become one when spatially varying surface fluxes are introduced.

**Recommended fix for future heterogeneous case:** Replace the scalar `lax.cond` with element-wise `jnp.where`:
```python
psi2D_m = jnp.where(is_stable_2D, psi_m_stable, psi_m_unstable)
```
where `is_stable_2D = invOB >= 0` is a 2D boolean field.

---

### 6.8 Design Issue: Statistics Capture Stale SGS Coefficients on Static-SGS Timesteps

**File:** `src/Main.py`, around line 326–333

When `dynamicSGS_call_time > 1`, the statistics call on static-SGS timesteps uses `Cs2_1D_avg1`, `Cs2_1D_avg2`, `Cs2PrRatio_1D`, `beta1_1D`, `beta2_1D` that were computed on the **previous dynamic timestep**. These are stale by up to `dynamicSGS_call_time` iterations. For the default `dynamicSGS_call_time = 1` this is not an issue since dynamic SGS runs every step. If the user sets `dynamicSGS_call_time > 1` to speed up the simulation, the reported Cs² profiles will reflect the last dynamic computation, not the current state.

This is arguably acceptable, but should be documented, and the statistics output could flag which steps are dynamic vs. static.

---

### 6.9 Minor: `SurfaceFlux_HomogeneousConstantFlux` Returns a Scalar for `qz_sfc_avg`

**File:** `src/surface/SurfaceFlux.py`, line 158–159

```python
qz_sfc_avg = jnp.mean(qz_sfc)
...
return M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions
```

The function returns `qz_sfc_avg` (a scalar) but `SurfaceFlux_HeterogeneousConstantFlux` also returns `qz_sfc_avg` (a scalar). Downstream, `Main.py` passes `qz_sfc` (the 2D array from `DerivedVars.py`) to `potentialTemperatureGradients` and SGS functions, not `qz_sfc_avg`. The returned `qz_sfc_avg` is only used for printing diagnostics:

```python
print(f"  Sensible Heat Flux:   {jnp.mean(qz_sfc):.2f} K m/s")
```

This is consistent but could be cleaner if the return value were used directly.

---

### 6.10 Minor: Top Boundary Condition for `u` and `v` is Implicit

**File:** `src/pde/NSE_TimeAdvancement.py`, lines 71–72

```python
u_new = u_new.at[:, :, nz - 1].set(u_new[:, :, nz - 2])
v_new = v_new.at[:, :, nz - 1].set(v_new[:, :, nz - 2])
```

This imposes a zero-gradient (Neumann) condition on `u` and `v` at the top. Combined with the Rayleigh damping layer that nudges velocities toward zero fluctuations, this is physically reasonable but is not documented anywhere in the code or docstrings. For simulations where the damping layer starts well below the domain top, the actual top-level values are just the damped result from the layer below, so the Neumann condition is inconsequential. But without the damping layer (`optDamping = 0`), this could allow unphysical behavior to accumulate at the top boundary.

---

## 7. Surface Layer Extension: Time-Varying Temperature and Fluxes

### 7.1 Motivation

The current `SurfaceFlux.py` supports only a prescribed, spatially uniform, time-invariant sensible heat flux `SensibleHeatFlux` (set in `Config.py`). Many ABL cases require:

- **Diurnal cycle (DC_WANGARA):** surface heat flux varies sinusoidally over 24 hours.
- **GABLS3:** a prescribed time series of surface temperature (not flux) drives the boundary layer.
- **SBL with prescribed cooling:** surface temperature decreases at a fixed rate.

Three distinct physical configurations should be supported:

| Mode | What is prescribed | What is computed |
|------|-------------------|-----------------|
| `optSfc = 0` (current) | Constant `qz_sfc` | `u*`, stability |
| `optSfc = 1` (new) | Time-varying `qz_sfc(t)` | `u*`, stability |
| `optSfc = 2` (new) | Time-varying `θ_s(t)` | `qz_sfc`, `u*`, stability |

### 7.2 What Changes in the Codebase

The key insight is that **`qz_sfc` is already passed as an argument** throughout the call graph. It is not captured from a global inside any JIT-compiled function. Specifically, it appears as a function argument in:

- `potentialTemperatureGradients(TH, TH_fft, kx2, ky2, ustar, qz_sfc, MOSTfunctions, ZeRo3D)`
- `DivFluxDynamicSGS(dynamicSGSmomentum[9:], TH, dTHdx, dTHdy, dTHdz, qz_sfc, ...)`
- `DivFluxStaticSGS(..., dTHdx, dTHdy, dTHdz, qz_sfc, ...)`

This means the change is **localized**: we only need to:
1. Replace the static `qz_sfc` from `DerivedVars.py` with a dynamically updated one in `Main.py`
2. Add the logic to compute the current `qz_sfc` at each timestep
3. Optionally add new surface flux functions for the prescribed-temperature mode

### 7.3 Mode 1: Time-Varying Prescribed Heat Flux

This is the simplest extension. The physical setup prescribes `H(t)` directly (e.g., a diurnal sine wave, or a step change).

**Changes to `Config.py`:**

Add a new option and time series specification:

```python
# Surface configuration (extended)
optSurfFlux = 1  # 0: constant flux, 1: time-varying flux, 2: prescribed temperature

# For optSurfFlux = 0 (existing)
SensibleHeatFlux = 0.0  # K m/s, constant

# For optSurfFlux = 1 (new)
# Option A: sinusoidal diurnal cycle
SurfaceFluxOption = 'sinusoidal'
SFC_amplitude = 0.1       # K m/s, amplitude
SFC_period = 86400.0      # s, period (one day)
SFC_phase = 0.0           # s, phase offset
SFC_mean = 0.0            # K m/s, mean value

# Option B: time series from file
# SurfaceFluxOption = 'timeseries'
# SurfaceFlux_file = 'input/surface_flux.txt'
# (file format: two columns: time[s]  flux[K m/s])
```

**Changes to `Main.py`:**

Before the main time loop, initialize `qz_sfc` as a mutable variable:

```python
# Initialize qz_sfc (will be updated each iteration for time-varying cases)
qz_sfc = qz_sfc_initial  # from DerivedVars as starting value
```

Inside the loop, compute the current flux before the surface flux call:

```python
if optSurfFlux == 1:  # time-varying flux
    t_current = (iteration - 1) * dt  # physical time in seconds
    qz_sfc_value = compute_flux_at_time(t_current)
    qz_sfc = qz_sfc_value * jnp.ones((nx, ny)) / (u_scale * TH_scale)
```

**New helper function (e.g., in `SurfaceFlux.py` or a new `SurfaceForcing.py`):**

```python
def compute_flux_at_time(t, option, amplitude, period, phase, mean):
    """
    Compute the surface heat flux (K m/s) at physical time t (seconds).
    This is a pure Python function - it returns a scalar that is then
    broadcast to a JAX array in Main.py.
    """
    if option == 'sinusoidal':
        return mean + amplitude * np.sin(2 * np.pi * (t - phase) / period)
    elif option == 'timeseries':
        return np.interp(t, flux_times, flux_values)
```

Note that this helper function is intentionally **not** JIT-compiled — it runs in Python and returns a scalar. The result is then wrapped in `jnp.ones((nx, ny))` to form the JAX array. This approach avoids needing to trace through the time-series interpolation logic.

**For a sinusoidal diurnal cycle, the JAX-native approach** would be cleaner if `t` is already a JAX scalar in the loop:

```python
qz_sfc_scalar = SFC_mean + SFC_amplitude * jnp.sin(
    2 * jnp.pi * (iteration * dt_nondim * z_scale / u_scale - SFC_phase) / SFC_period
)
qz_sfc = qz_sfc_scalar * jnp.ones((nx, ny))
```

This is fully JAX-native and JIT-safe since all operations are on scalars and the result is a concrete array.

### 7.4 Mode 2: Prescribed Surface Temperature (Bulk Formula)

This is required for cases like GABLS1 (prescribed surface cooling rate) and GABLS3 (prescribed surface temperature time series). The surface heat flux is no longer prescribed but is computed from a bulk formula:

```
qz_sfc(i,j) = −Ch · u*(i,j) · [θ(z₁, i,j) − θ_s(t)]
```

where `Ch` is the heat transfer coefficient (analogous to `u*²/M` for momentum), and `θ_s(t)` is the prescribed surface temperature.

This introduces a **two-way coupling** between the surface temperature and the flux — the flux depends on both `u*` and the temperature difference, and `u*` depends on the flux through the Obukhov length. This requires a new iterative surface flux routine.

**Changes to `Config.py`:**

```python
optSurfFlux = 2  # prescribed surface temperature

# For optSurfFlux = 2 (new)
T_surface_initial = 265.0   # K, initial surface temperature
T_surface_cooling = -0.25   # K/hr, cooling rate (GABLS1: -0.25 K/hr)
# or for a file:
# T_surface_file = 'input/T_surface.txt'
```

**New surface flux function (in `SurfaceFlux.py`):**

The critical design decision is whether to iterate within the surface flux function (solving for consistent `u*`, `L`, and `qz_sfc`) or to use a lagged single-pass approach as the existing code does. Given that the existing code is already first-order in time for the stability coupling, a single-pass approach is consistent:

```python
def SurfaceFlux_PrescribedTemperature(u, v, TH, theta_surface, MOSTfunctions):
    """
    Compute surface fluxes when the surface temperature is prescribed.
    Uses a single-pass update: stability from previous step, then new u*, then new flux.
    
    Parameters
    ----------
    theta_surface : scalar or 2D array
        Prescribed surface potential temperature (non-dimensional)
    """
    # Unpack previous stability functions
    (psi2D_m, psi2D_m0, psi2D_h, psi2D_h0, fi2D_m, fi2D_h) = MOSTfunctions

    # Near-surface wind speed (local for heterogeneous)
    M_sfc_loc = jnp.sqrt((u[:, :, 0] + Ugal) ** 2 + v[:, :, 0] ** 2)

    # Near-surface temperature
    TH_sfc_loc = TH[:, :, 0]

    # Friction velocity (from log-law + previous stability)
    denom_m = jnp.log(0.5 * dz * z_scale / z0m) + psi2D_m - psi2D_m0
    ustar = vonk * M_sfc_loc / denom_m

    # Heat transfer coefficient (from log-law for scalars + previous stability)
    denom_h = jnp.log(0.5 * dz * z_scale / z0T) + psi2D_h - psi2D_h0
    Ch = vonk * ustar / denom_h   # = vonk^2 * M / (denom_m * denom_h)

    # Surface heat flux from bulk formula
    qz_sfc_2D = -Ch * (TH_sfc_loc - theta_surface)

    # Obukhov length and new stability functions (same as current code)
    qz_sfc_avg = jnp.mean(qz_sfc_2D)
    invOB = -(vonk * g_nondim * qz_sfc_avg) / (
        jnp.maximum(ustar, 1e-6) ** 3 * jnp.mean(TH_sfc_loc)
    )
    is_stable = invOB >= 0

    # ... (update psi, fi as in current code) ...

    return M_sfc_loc, ustar, qz_sfc_2D, jnp.mean(qz_sfc_2D), invOB, MOSTfunctions
```

The key difference from Mode 1 is that `qz_sfc` is now a **2D array** computed each timestep from the bulk formula, not a prescribed scalar broadcast to 2D. This means the downstream functions (`potentialTemperatureGradients`, `DivFluxDynamicSGS`, etc.) receive a spatially varying flux, which they already support — their interfaces accept `qz_sfc` as an `(nx, ny)` array.

**Changes to `Main.py`:**

```python
if optSurfFlux == 0:  # constant prescribed flux (existing)
    (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
        SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions))
    # qz_sfc stays as initialized from DerivedVars.py

elif optSurfFlux == 1:  # time-varying prescribed flux (new)
    t_current = (iteration - 1) * dt
    qz_sfc = compute_qz_sfc_at_time(t_current)  # returns (nx, ny) array
    (M_sfc_loc, ustar, qz_sfc_avg, invOB, MOSTfunctions) = (
        SurfaceFlux_HomogeneousConstantFlux(u, v, TH, MOSTfunctions))
    # Note: SurfaceFlux_Homogeneous uses jnp.mean(qz_sfc) internally,
    # which will now read the updated qz_sfc

elif optSurfFlux == 2:  # prescribed surface temperature (new)
    theta_surface = compute_theta_surface_at_time(t_current)  # scalar, non-dimensional
    (M_sfc_loc, ustar, qz_sfc, qz_sfc_avg, invOB, MOSTfunctions) = (
        SurfaceFlux_PrescribedTemperature(u, v, TH, theta_surface, MOSTfunctions))
```

### 7.5 The `qz_sfc` Variable in `DerivedVars.py`

Currently `qz_sfc` is defined globally:

```python
qz_sfc = SensibleHeatFlux * jnp.ones((nx, ny)) / (u_scale * TH_scale)
```

This global definition is imported by `Main.py` via `from .config.DerivedVars import *`. For the time-varying extension, we should rename this to `qz_sfc_initial` and use it only to initialize the variable in `Main.py`:

```python
# In DerivedVars.py:
qz_sfc_initial = SensibleHeatFlux * jnp.ones((nx, ny)) / (u_scale * TH_scale)

# In Main.py (initialization section):
qz_sfc = qz_sfc_initial.copy()   # will be updated each timestep for optSurfFlux > 0
```

This change is backward-compatible: for `optSurfFlux = 0`, `qz_sfc` is never updated in the loop and remains equal to `qz_sfc_initial`.

### 7.6 JAX Compatibility Notes

Several JAX-specific issues arise when implementing time-varying surface forcing:

**Issue 1: Python-level conditionals in the time loop.**  
The `if optSurfFlux == 1:` branch in `Main.py` is a Python-level conditional, not a JAX `lax.cond`. This is correct — `Main.py` is not JIT-compiled. The decision about which surface flux function to call is made in Python at runtime. The called function itself is JIT-compiled.

**Issue 2: Updating `qz_sfc` each iteration.**  
`qz_sfc` is a JAX array that is created once and then replaced at each timestep (not mutated in-place). JAX arrays are immutable; `qz_sfc = new_value` replaces the Python reference. This is fine since `qz_sfc` is passed by reference into JIT functions which receive a concrete array value at trace time. A new trace is triggered only if the array's **shape or dtype** changes, not its values. Since `qz_sfc` always has shape `(nx, ny)` and the same dtype, there is no recompilation overhead.

**Issue 3: `jax.lax.cond` for stability branch with 2D `invOB`.**  
When `optSurfFlux = 2`, `invOB` is a 2D array (different stability at different points). The current `jax.lax.cond(is_stable, ...)` with a scalar condition must be replaced with element-wise operations:
```python
# Replace scalar lax.cond with element-wise jnp.where:
psi2D_m = jnp.where(invOB >= 0, 5.0 * z1_over_L, psi_unstable_m)
fi2D_m  = jnp.where(invOB >= 0, 1.0 + 5.0 * z1_over_L, fi_unstable_m)
```
This is a necessary change for the heterogeneous, prescribed-temperature case. The stable and unstable branches must both be **evaluated everywhere** (JAX does not short-circuit), and `jnp.where` selects the correct one per point. This is acceptable since the stability functions are cheap to evaluate.

**Issue 4: `compute_theta_surface_at_time` should be Python, not JAX.**  
The surface temperature schedule (linear cooling, sinusoidal, or file interpolation) is best computed in Python and the result passed as a JAX scalar. This avoids tracing through NumPy interpolation or conditional logic that JAX cannot compile efficiently.

### 7.7 Summary of Required Changes

For Mode 1 (time-varying prescribed flux):

| File | Change |
|------|--------|
| `Config.py` | Add `optSurfFlux` values, time series parameters |
| `DerivedVars.py` | Rename `qz_sfc` → `qz_sfc_initial` |
| `Main.py` | Initialize `qz_sfc = qz_sfc_initial.copy()`; update it each iteration for `optSurfFlux = 1` |
| `SurfaceFlux.py` | No change needed (reads `qz_sfc` from arg `jnp.mean(qz_sfc)`) |

For Mode 2 (prescribed surface temperature):

| File | Change |
|------|--------|
| `Config.py` | Add `optSurfFlux = 2`, temperature schedule parameters |
| `DerivedVars.py` | Rename `qz_sfc` → `qz_sfc_initial` |
| `Main.py` | Handle `optSurfFlux = 2` branch; accept 2D `qz_sfc` back from surface function |
| `SurfaceFlux.py` | Add `SurfaceFlux_PrescribedTemperature` function; replace scalar `lax.cond` with element-wise `jnp.where` for 2D stability |

The changes are deliberately minimal and backward-compatible. Existing `Config.py` files with `optSurfFlux = 0` continue to work without modification.

---

## 8. Running the Code: Workflow and I/O Management

### 8.1 Directory Layout

Each run directory is self-contained: it holds the configuration (`Config.py`),
the input-creation script (`CreateInputs_*.py`), the initial-condition files
(`input/`), and the simulation output (`output/`). The source code in `src/` is
never modified per run.

```
JAXALFA0.1/
├── src/
│   ├── config/
│   │   ├── ConfigLoader.py    ← loads Config.py from JAXALFA_RUNDIR (mandatory)
│   │   └── DerivedVars.py
│   ├── initialization/
│   │   └── Initialization.py  ← reads from JAXALFA_RUNDIR/input/
│   └── Main.py                ← writes to JAXALFA_RUNDIR/output/
└── examples/
    └── CBL_N91/
        └── runs/
            └── 128x128x128/
                ├── Config.py                   ← run-specific parameters
                ├── CreateInputs_CBL128_N91.py  ← generates initial conditions
                ├── input/
                │   ├── vel.ini
                │   └── TH.ini
                └── output/
                    └── ALFA_Statistics_*.npz
```

`JAXALFA_RUNDIR` **must** be set before launching JAX-ALFA. It is the single
pointer that tells every module where to find `Config.py`, `input/`, and
`output/`. The code raises `EnvironmentError` immediately at import time if it
is not set.

### 8.2 Step-by-Step Run Procedure

**Step 1 — Set the run directory**

```bash
export JAXALFA_RUNDIR=/path/to/examples/CBL_N91/runs/128x128x128
```

**Step 2 — Generate initial conditions**

```bash
python $JAXALFA_RUNDIR/CreateInputs*.py
```

This writes `vel.ini` and `TH.ini` to `$JAXALFA_RUNDIR/input/`, resolved
relative to the script's own location regardless of the current working
directory.

**Step 3 — Run the simulation from the package root**

```bash
cd /path/to/JAXALFA0.1
python -m src.Main
```

JAX-ALFA must be run as a package (`python -m src.Main`) rather than as a plain
script (`python src/Main.py`) because all imports use relative paths (`from .config
import ...`). Running as a plain script breaks the package hierarchy.

Outputs land directly in `$JAXALFA_RUNDIR/output/` and are never automatically
deleted.

### 8.3 Restart Runs

For a restart, two things must change:

1. Set `istep` in `$JAXALFA_RUNDIR/Config.py` to the iteration number to resume
   from (e.g., `istep = 25201` if the last checkpoint was at iteration 25200).
2. Replace `vel.ini` and `TH.ini` in `$JAXALFA_RUNDIR/input/` with fields
   extracted from the last `ALFA_3DFields_Iteration_N.npz` checkpoint. A helper
   script is needed to convert the `.npz` arrays back to the column-vector text
   format that `Initialization.py` expects (Fortran column-major order,
   space-separated).

There is currently no built-in restart helper. Writing one is straightforward:

```python
import numpy as np
import os

rundir = os.environ['JAXALFA_RUNDIR']
data = np.load(os.path.join(rundir, 'output', 'ALFA_3DFields_Iteration_25200.npz'))
u = data['u']   # shape (nx, ny, nz), already has Ugal added back
v = data['v']
w = data['w']

vel = np.column_stack([u.reshape(-1, order='F'),
                       v.reshape(-1, order='F'),
                       w.reshape(-1, order='F')])
input_dir = os.path.join(rundir, 'input')
np.savetxt(os.path.join(input_dir, 'vel.ini'), vel)
np.savetxt(os.path.join(input_dir, 'TH.ini'), data['TH'].reshape(-1, order='F'))
```

---

## 9. Input/Output Folder Strategy

Each run directory is the authoritative unit of work. The configuration, initial
conditions, and outputs all live together, and `JAXALFA_RUNDIR` is the sole
mechanism for telling the source code where to find them.

### 9.1 Layout: Run Directory as the Unit of Work

```
JAXALFA0.1/
├── src/                       ← source code only, never modified per run
└── examples/
    └── CBL_N91/
        └── runs/
            └── 128x128x128/
                ├── Config.py               ← run parameters (authoritative)
                ├── CreateInputs_*.py       ← initial condition generator
                ├── input/
                │   ├── vel.ini
                │   └── TH.ini
                └── output/                 ← simulation writes directly here
```

### 9.2 `JAXALFA_RUNDIR` Is Mandatory

`JAXALFA_RUNDIR` is not optional. The code raises `EnvironmentError` at import
time if it is unset, with a message showing the correct usage:

```
JAXALFA_RUNDIR is not set.
Set it to the run directory before launching JAX-ALFA:

    export JAXALFA_RUNDIR=/path/to/run_directory
    python $JAXALFA_RUNDIR/CreateInputs*.py
    python -m src.Main
```

This eliminates the ambiguity of the previous if/else fallback (legacy mode),
which allowed the code to run silently with stale or wrong inputs.

### 9.3 Output Protection: Never Auto-Delete

The `shutil.rmtree(OutputDir)` that existed in the previous version of `Main.py`
has been removed. `OutputDir` is now always `$JAXALFA_RUNDIR/output/`, created
with `exist_ok=True`. Output files from previous runs in the same directory are
never deleted automatically.

If outputs need to be cleared before a fresh run, do so explicitly:

```bash
rm -rf $JAXALFA_RUNDIR/output/
```

### 9.4 `CreateInputs` Convention

Each `CreateInputs_*.py` script lives at the run directory root (alongside
`Config.py`) and writes initial conditions to `input/` relative to its own
location:

```python
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(input_dir, exist_ok=True)
np.savetxt(os.path.join(input_dir, 'vel.ini'), vel_data)
np.savetxt(os.path.join(input_dir, 'TH.ini'), TH_flat)
```

This makes the script runnable from any working directory — the `__file__`-based
path is independent of the current working directory.

### 9.5 Summary

| Before | After |
|--------|-------|
| `JAXALFA_RUNDIR` optional; legacy mode silently used `__file__` paths | `JAXALFA_RUNDIR` mandatory; unset raises `EnvironmentError` immediately |
| `src/config/Config.py` held hardcoded defaults + optional env-var override | `src/config/ConfigLoader.py` loads only from `JAXALFA_RUNDIR/Config.py` |
| `Main.py` wiped `output/` on every startup | `output/` is never auto-deleted |
| `CreateInputs_*.py` lived in `codes/` subdirectory | `CreateInputs_*.py` lives at run directory root |
| Manual 5-step copy workflow | Two-command launch: `export` + `python -m src.Main` |

---

## 10. Code Changes Implemented

The following changes have been applied to implement the run-directory strategy
described in Section 9. All changes are in effect as of 2026-05-01.

### 10.1 `src/config/Config.py` → `src/config/ConfigLoader.py`

`src/config/Config.py` has been renamed to `src/config/ConfigLoader.py` to
distinguish it from the per-run `Config.py` files that live in each run
directory. The new file contains no hardcoded defaults. It reads
`JAXALFA_RUNDIR` from the environment (mandatory) and executes
`$JAXALFA_RUNDIR/Config.py`:

```python
_rundir = os.environ.get('JAXALFA_RUNDIR')
if _rundir is None:
    raise EnvironmentError(
        "\n\nJAXALFA_RUNDIR is not set.\n"
        "Set it to the run directory before launching JAX-ALFA:\n\n"
        "    export JAXALFA_RUNDIR=/path/to/run_directory\n"
        "    python $JAXALFA_RUNDIR/CreateInputs*.py\n"
        "    python -m src.Main\n"
    )

with open(os.path.join(_rundir, 'Config.py')) as _f:
    exec(_f.read())
```

`src/config/Config.py` has been deleted.

All source files that previously imported `from ..config.Config import *` have
been updated to `from ..config.ConfigLoader import *`. Files that used
`from ..config import Config` now use `from ..config import ConfigLoader as Config`.

### 10.2 `src/initialization/Initialization.py`

The `InputDir` if/else block (which provided a fallback to `__file__`-relative
path when `JAXALFA_RUNDIR` was unset) has been removed. `InputDir` is now set
directly and unconditionally from the environment variable:

```python
InputDir = os.path.join(os.environ['JAXALFA_RUNDIR'], 'input')
```

If `JAXALFA_RUNDIR` is not set, `ConfigLoader.py` raises `EnvironmentError`
at import time before this line is reached.

### 10.3 `src/Main.py`

The `OutputDir` if/else block and the `shutil.rmtree` call have been removed.
`OutputDir` is now always `$JAXALFA_RUNDIR/output/`, created with
`exist_ok=True`:

```python
OutputDir = os.path.join(os.environ['JAXALFA_RUNDIR'], 'output')
os.makedirs(OutputDir, exist_ok=True)
```

The `import shutil` statement has also been removed since it is no longer used.

### 10.4 `CreateInputs_*.py` Scripts Moved to Run Directory Root (13 files)

All 13 input-creation scripts have been moved from `codes/` to the run directory
root and their `input_dir` path updated to remove the `'..'` level:

```python
# Before (in codes/ subdirectory):
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'input')

# After (at run directory root):
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
```

Files moved:

| Example | Resolution | File |
|---------|-----------|------|
| CBL_N91 | 32×16×24 | `CreateInputs_CBL32_N91.py` |
| CBL_N91 | 32×32×32 | `CreateInputs_CBL32_N91.py` |
| CBL_N91 | 64×64×64 | `CreateInputs_CBL64_N91.py` |
| CBL_N91 | 96×96×96 | `CreateInputs_CBL96_N91.py` |
| CBL_N91 | 128×128×128 | `CreateInputs_CBL128_N91.py` |
| CBL_N91 | 192×192×192 | `CreateInputs_CBL192_N91.py` |
| CBL_N91 | 256×256×256 | `CreateInputs_CBL256_N91.py` |
| CBL_N91_G3 | 256×256×256 | `CreateInputs_CBL256_N91_G3.py` |
| CBL_N91_G5 | 256×256×256 | `CreateInputs_CBL256_N91_G5.py` |
| NBL_A94 | 40×40×40 | `CreateInputs_NBL40_A94.py` |
| NBL_A94 | 64×64×64 | `CreateInputs_NBL64_A94.py` |
| NBL_A94 | 80×80×80 | `CreateInputs_NBL80_A94.py` |
| NBL_A94 | 128×128×128 | `CreateInputs_NBL128_A94.py` |

The original scripts in `codes/` are retained unchanged for reference.

### 10.5 `Config.py` at Each Run Directory Root (13 files, unchanged)

The per-run `Config.py` files created in the previous round of changes remain
at the root of every run directory. No changes to these files were needed.

### 10.6 End-to-End Workflow

Running a case requires two commands from any working directory:

```bash
export JAXALFA_RUNDIR=/path/to/examples/CBL_N91/runs/128x128x128
python $JAXALFA_RUNDIR/CreateInputs*.py
python -m src.Main
```

Step 1 (`export`) sets the run directory. Step 2 runs the single
`CreateInputs_*.py` script found there — the shell glob expands to exactly one
file, so no filename needs to be typed. Step 3 runs the simulation; outputs land
in `$JAXALFA_RUNDIR/output/` and are never automatically deleted.

---

*End of Report*
