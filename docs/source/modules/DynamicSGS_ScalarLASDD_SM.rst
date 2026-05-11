SGS Model: LASDD-SM (Scalar)
============================

Scalar (potential temperature) Locally-Averaged Scale-Dependent Dynamic
SGS model using the Smagorinsky base formulation. Called for ``optSgs = 1``
(LASDD-SM) and ``optSgs = 3`` (LAD-SM). For LAD variants (``optSgs = 3``),
the scalar scale-dependence parameter ``beta2`` is set to 1.

.. automodule:: src.subgridscale.DynamicSGS_ScalarLASDD_SM
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: DynamicSGS_ScalarLASDD_SM
---------------------------------------

.. literalinclude:: ../../../src/subgridscale/DynamicSGS_ScalarLASDD_SM.py
   :language: python
   :linenos:
   :caption: DynamicSGS_ScalarLASDD_SM.py
