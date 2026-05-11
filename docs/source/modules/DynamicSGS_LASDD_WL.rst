SGS Model: LASDD-WL (Momentum)
==============================

Locally-Averaged Scale-Dependent Dynamic SGS model using the Wong-Lilly
base formulation. Called for ``optSgs = 2`` (LASDD-WL) and ``optSgs = 4``
(LAD-WL). For LAD variants (``optSgs = 4``), the scale-dependence parameter
``beta`` is set to 1 rather than computed.

.. automodule:: src.subgridscale.DynamicSGS_LASDD_WL
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: DynamicSGS_LASDD_WL
---------------------------------

.. literalinclude:: ../../../src/subgridscale/DynamicSGS_LASDD_WL.py
   :language: python
   :linenos:
   :caption: DynamicSGS_LASDD_WL.py
