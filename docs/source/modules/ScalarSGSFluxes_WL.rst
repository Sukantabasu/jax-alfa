Scalar SGS Flux Computations: Wong-Lilly
=========================================

Computes SGS scalar (potential temperature) fluxes using the Wong-Lilly
(WL) base formulation:

.. math::

   q_i = -\frac{C_{WL}}{Pr_t} \Delta^{4/3} \frac{\partial \bar{\theta}}{\partial x_i}

Note the absence of the strain rate magnitude factor compared to the
Smagorinsky formulation. Used for ``optSgs = 2`` (LASDD-WL) and
``optSgs = 4`` (LAD-WL).

.. automodule:: src.subgridscale.ScalarSGSFluxes_WL
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: ScalarSGSFluxes_WL
---------------------------------

.. literalinclude:: ../../../src/subgridscale/ScalarSGSFluxes_WL.py
   :language: python
   :linenos:
   :caption: ScalarSGSFluxes_WL.py
