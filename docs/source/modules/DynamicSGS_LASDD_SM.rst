SGS Model: LASDD-SM (Momentum)
==============================

Locally-Averaged Scale-Dependent Dynamic SGS model using the Smagorinsky
base formulation. Called for ``optSgs = 1`` (LASDD-SM) and ``optSgs = 3``
(LAD-SM). For LAD variants (``optSgs = 3``), the scale-dependence parameter
``beta`` is set to 1 rather than computed.

.. automodule:: src.subgridscale.DynamicSGS_LASDD_SM
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: DynamicSGS_LASDD_SM
---------------------------------

.. literalinclude:: ../../../src/subgridscale/DynamicSGS_LASDD_SM.py
   :language: python
   :linenos:
   :caption: DynamicSGS_LASDD_SM.py
