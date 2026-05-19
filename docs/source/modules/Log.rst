Change Log
==========

JAX-ALFA 0.1.2 (May 19, 2026)
------------------------------

* New Features

  * **Prognostic specific humidity** (``optMoisture``, ``optMoistureSurfBC``):
    specific humidity *Q* is now a fully coupled prognostic scalar — transport,
    surface boundary condition (constant flux, time-varying flux, or
    time-varying prescribed *Q*), large-scale moisture advection
    (``optAdvection``), and Rayleigh damping all support *Q*.

  * **Virtual potential temperature buoyancy** (``BuoyancyOpt1``,
    ``BuoyancyOpt2``): buoyancy now uses
    *θ*\ :sub:`v` = (*θ* + *T*\ :sub:`0`) × (1 + 0.61 *Q*) so that
    the moisture loading term is *O*(2 K), not *O*(0 K).

  * **GABLS3 case study** (``examples/SBL_GABLS3``): complete 128\ :sup:`3`
    configuration with time- and height-varying geostrophic winds, temperature
    and moisture large-scale advection, and observed 0.25 m surface temperature
    and humidity boundary conditions.  Input generation scripts
    (``CreateInputs_GABLS3.py``, ``CreateSurfaceBC_GABLS3.py``,
    ``CreateGeoWind_GABLS3.py``, ``CreateAdvForcing_GABLS3.py``) are included.

  * **Time–height diagnostic notebook** for GABLS3
    (``SBL_GABLS3_TimeHeight.ipynb``): wind speed, potential temperature,
    specific humidity, velocity variances, and turbulent flux profiles over the
    9-hour simulation window.

  * **GPU selection** (``GPU_ID`` in ``Config.py``): each run directory can
    now specify which GPU to use on a multi-GPU workstation.  On SLURM clusters
    the environment variable ``CUDA_VISIBLE_DEVICES`` set by the scheduler
    takes precedence automatically.

-------

* Bug Fixes

  * Fixed ``BuoyancyOpt2`` denominator: was dividing by
    *θ*\ :sub:`v,bar` + *T*\ :sub:`0`; now divides by *θ*\ :sub:`v,bar`
    directly (which already carries absolute temperature magnitude).

  * Fixed large-scale advection jump timing in ``CreateAdvForcing`` scripts:
    the "before-jump" time was offset one step too late (``t_jump + ε``);
    corrected to ``t_jump − ε``.

  * Fixed latent ``NameError``: ``Initialize_AdvForcing`` was missing from
    ``Imports.py``; any run with ``optAdvection ≥ 1`` would crash at startup.

  * Fixed ``run_simulation.sh`` and ``run_simulation_dgx.sh``: input-generation
    scripts (``CreateGeoWind*``, ``CreateAdvForcing*``) were not executed before
    the main solver; they are now run in the correct order.

-------

* Changes

  * ``Initialize_AdvForcing()`` now returns a 4-tuple
    ``(AdvForcing_U, AdvForcing_V, AdvForcing_TH, AdvForcing_Q)``.
    If ``Qadv_series`` is absent from the ``.npz`` file, *Q* advection
    silently defaults to zero (backward-compatible with dry runs).

  * ``ConfigLoader.py``: added backward-compatible default ``GPU_ID = 0`` so
    existing ``Config.py`` files that omit the key do not crash.

-------

JAX-ALFA 0.1.1 (May 10, 2025)
------------------------------

* New Features

  * Added four SGS model options via ``optSgs`` in ``Config.py``:

    * ``optSgs = 1``: LASDD-SM — Locally-Averaged Scale-Dependent Dynamic,
      Smagorinsky base model (scale-dependence parameter ``beta`` computed)
    * ``optSgs = 2``: LASDD-WL — Locally-Averaged Scale-Dependent Dynamic,
      Wong-Lilly base model (``beta`` computed)
    * ``optSgs = 3``: LAD-SM — Locally-Averaged Dynamic, Smagorinsky
      (``beta = 1``, polynomial root-finding skipped)
    * ``optSgs = 4``: LAD-WL — Locally-Averaged Dynamic, Wong-Lilly
      (``beta = 1``, polynomial root-finding skipped)

  * Added ``Cwl`` and ``CwlPrRatio`` initialization coefficients in
    ``Config.py`` for WL model variants (``optSgs = 2, 4``).

-------

* Bug Fixes

  * Fixed dynamic SGS coefficient caching when ``dynamicSGS_call_time > 1``:
    ``Cs2_3D`` and ``Cs2PrRatio_3D`` are now updated at each dynamic call
    step and reused on non-call steps, replacing the static initialization
    values that were previously kept unchanged.

  * Fixed cached-step stress and flux computation for WL variants: the
    static SGS path (used on non-call steps) now dispatches to Wong-Lilly
    stress and flux functions when ``optSgs in [2, 4]``, rather than
    incorrectly applying the Smagorinsky formula.

-------

* Changes

  * ``DynamicSGS_LASDD_SM.py``, ``DynamicSGS_LASDD_WL.py``,
    ``DynamicSGS_ScalarLASDD_SM.py``, ``DynamicSGS_ScalarLASDD_WL.py``:
    added ``computeBeta`` flag — ``True`` for LASDD variants (``optSgs = 1, 2``),
    ``False`` for LAD variants (``optSgs = 3, 4``).

  * ``DynamicSGS_Main.py``: dispatch conditions updated from ``optSgs == 2/3``
    to ``optSgs in [1, 3]`` (SM) and ``optSgs in [2, 4]`` (WL).

  * ``Main.py``: dynamic SGS condition changed from ``optSgs >= 2`` to
    ``optSgs >= 1`` to include the new ``optSgs = 1`` option.

  * ``StaticSGS_Main.py``: added SM/WL dispatch so cached steps apply
    the correct stress and flux formulas for each model family.

  * ``DerivedVars.py``: initial ``Cs2_3D`` / ``Cs2PrRatio_3D`` fields now
    branch on model family — WL models use ``Cwl`` / ``CwlPrRatio``;
    SM models use ``Cs2`` / ``Cs2PrRatio``.

  * All ``Config.py`` examples: ``optSgs = 2`` (old LASDD-SM) changed to
    ``optSgs = 1`` (new LASDD-SM numbering); ``optSgs = 0`` (Static)
    removed from user-facing options.

-------

* Deprecations

  * ``optSgs = 0`` (Static Smagorinsky) is no longer a documented user
    option. The static SGS path is retained internally as the caching
    mechanism for dynamic models when ``dynamicSGS_call_time > 1``.

-------

JAX-ALFA 0.1 (April 29, 2025)
------------------------------

* New Features

-------

* Bug Fixes

-------

* Changes

-------

* Deprecations
