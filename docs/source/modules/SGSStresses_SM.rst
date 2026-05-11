SGS Stress Computations: Smagorinsky
=====================================

Computes SGS momentum stresses using the Smagorinsky (SM) base formulation:

.. math::

   \tau_{ij} = -2 C_s^2 \Delta^2 |\bar{S}| \bar{S}_{ij}

Used for ``optSgs = 1`` (LASDD-SM) and ``optSgs = 3`` (LAD-SM), both on
dynamic call steps and on cached (non-call) steps when
``dynamicSGS_call_time > 1``.

.. automodule:: src.subgridscale.SGSStresses_SM
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: SGSStresses_SM
----------------------------

.. literalinclude:: ../../../src/subgridscale/SGSStresses_SM.py
   :language: python
   :linenos:
   :caption: SGSStresses_SM.py
