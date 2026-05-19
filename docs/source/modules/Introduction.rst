Introduction
============

JAX-ALFA is a JAX-based large-eddy simulation framework for atmospheric
boundary layer simulations. It leverages JAX's CPU/GPU/TPU acceleration
capabilities to provide highly efficient cross-platform simulations
without any code changes.

Features
--------

- Incompressible flow solver
- Spectral methods for horizontal derivatives
- Finite difference methods for vertical derivatives
- FFT-based direct Poisson solver
- Dynamic SGS coefficient computation
- Prognostic potential temperature and specific humidity scalars
- Virtual potential temperature buoyancy coupling
- JAX-accelerated computations for CPUs & GPUs
- Either single or double precision computations
