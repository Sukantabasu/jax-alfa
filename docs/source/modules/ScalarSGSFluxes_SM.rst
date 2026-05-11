Scalar SGS Flux Computations: Smagorinsky
==========================================

Computes SGS scalar (potential temperature) fluxes using the Smagorinsky
(SM) base formulation:

.. math::

   q_i = -2 \frac{C_s^2}{Pr_t} \Delta^2 |\bar{S}| \frac{\partial \bar{\theta}}{\partial x_i}

Used for ``optSgs = 1`` (LASDD-SM) and ``optSgs = 3`` (LAD-SM).

.. automodule:: src.subgridscale.ScalarSGSFluxes_SM
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: ScalarSGSFluxes_SM
---------------------------------

.. literalinclude:: ../../../src/subgridscale/ScalarSGSFluxes_SM.py
   :language: python
   :linenos:
   :caption: ScalarSGSFluxes_SM.py
