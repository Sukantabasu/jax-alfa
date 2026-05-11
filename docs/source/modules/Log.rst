Change Log
==========

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
