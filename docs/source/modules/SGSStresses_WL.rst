SGS Stress Computations: Wong-Lilly
=====================================

Computes SGS momentum stresses using the Wong-Lilly (WL) base formulation:

.. math::

   \tau_{ij} = -2 C_{WL} \Delta^{4/3} \bar{S}_{ij}

Note the absence of the strain rate magnitude factor compared to the
Smagorinsky formulation. Used for ``optSgs = 2`` (LASDD-WL) and
``optSgs = 4`` (LAD-WL), both on dynamic call steps and on cached
(non-call) steps when ``dynamicSGS_call_time > 1``.

.. automodule:: src.subgridscale.SGSStresses_WL
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: SGSStresses_WL
----------------------------

.. literalinclude:: ../../../src/subgridscale/SGSStresses_WL.py
   :language: python
   :linenos:
   :caption: SGSStresses_WL.py
