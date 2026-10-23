Pressure Terms (Thomas Solver)
==============================

Tridiagonal (Thomas algorithm) pressure solver, selected with
``optPressureSolver = 1`` in ``Config.py`` (used by all shipped cases).
The dense LU solver (``optPressureSolver = 0``) is described in
:doc:`NSE_PressureTerms`.

.. automodule:: src.pde.NSE_PressureTerms_Thomas
   :members:
   :undoc-members:
   :show-inheritance:

Source Code: NSE_PressureTerms_Thomas
-------------------------------------

.. literalinclude:: ../../../src/pde/NSE_PressureTerms_Thomas.py
   :language: python
   :linenos:
   :caption: NSE_PressureTerms_Thomas.py
