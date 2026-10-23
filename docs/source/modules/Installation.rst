Installation
============

Requirements
------------

- Python 3.10 or later
- `JAX <https://github.com/google/jax>`_ 0.4 or later
- NumPy
- SciPy

GPU execution additionally requires a CUDA-capable NVIDIA GPU with the
CUDA-enabled JAX build (see below).

Getting the Code
----------------

Clone the repository from GitHub::

    git clone https://github.com/Sukantabasu/jax-alfa.git
    cd jax-alfa

No compilation or ``pip install`` step is required.  JAX-ALFA is run
directly as a Python package from the repository root.

Installing JAX
--------------

JAX installation depends on your hardware platform.

**CPU only**::

    pip install -U jax

**GPU (NVIDIA CUDA)**::

    pip install -U "jax[cuda13]"
    # or, on CUDA 12 systems:
    pip install -U "jax[cuda12]"

For other platforms or CUDA versions, consult the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

Installing Other Dependencies
-----------------------------

::

    pip install numpy scipy

Verifying the Installation
--------------------------

From the repository root, run::

    python -c "import jax; print(jax.__version__); print(jax.devices())"

A GPU build should report a CUDA device; a CPU build will report a CPU device.

Running a Simulation
--------------------

Set the environment variable ``JAXALFA_RUNDIR`` to point to a run
directory, then launch the solver::

    export JAXALFA_RUNDIR=/path/to/jax-alfa/examples/SBL_GABLS1/runs/64x64x64_LASDD_SM_SP
    python $JAXALFA_RUNDIR/CreateInputs_SBL_GABLS1.py
    python $JAXALFA_RUNDIR/CreateSurfaceBC_GABLS1.py
    python -m src.Main

Each run directory provides the ``Create*.py`` scripts it needs (initial
conditions, and where applicable surface boundary conditions, geostrophic
wind and large-scale advection forcing); run all of them before the solver.

Always run ``python -m src.Main`` from the repository root so that the
``src`` package is importable.

Alternatively, use the provided convenience script, which deletes any previous
output of that run directory, runs all of its ``Create*.py`` scripts and then the
solver, writing ``run.log`` in the run directory::

    bash scripts/run_simulation.sh examples/SBL_GABLS1/runs/64x64x64_LASDD_SM_SP

Output files are written to ``$JAXALFA_RUNDIR/output/`` as compressed
NumPy archives (``*.npz``).  See the :doc:`Tutorial </tutorial/Tutorial_notebook>`
for a step-by-step walkthrough of the GABLS1 case.

Selecting CPU or GPU
--------------------

Set ``optGPU`` in the run directory's ``Config.py``::

    optGPU = 0   # CPU
    optGPU = 1   # GPU

On a workstation with multiple GPUs, set ``GPU_ID`` to select which
device to use::

    GPU_ID = 0   # first GPU
    GPU_ID = 1   # second GPU

On SLURM clusters the scheduler sets ``CUDA_VISIBLE_DEVICES``
automatically; ``GPU_ID`` is ignored in that case.

Running on a SLURM Cluster
--------------------------

A ready-to-use SLURM batch script ``scripts/run_simulation_dgx.sh`` is included
in the repository.  Before submitting, edit the following items inside the
script:

1. **Path variables** — set ``JAXALFA_ROOT`` and ``JAXALFA_RUNDIR``::

       export JAXALFA_ROOT=/path/to/jax-alfa
       export JAXALFA_RUNDIR=$JAXALFA_ROOT/examples/SBL_GABLS1/runs/...

2. **Conda initialisation** — update the ``source`` path to your cluster's
   conda installation::

       source /path/to/anaconda3/etc/profile.d/conda.sh

3. **Conda environment** — update the ``conda activate`` line to your
   JAX-enabled environment::

       conda activate /path/to/anaconda3/envs/jax-gpu

4. **SBATCH directives** — adjust the time limit, memory, and GPU count at
   the top of the script (and, if your cluster requires them, the account,
   partition and QoS) to match your cluster's configuration.

Then submit from the repository root with::

    mkdir -p slurmout
    sbatch scripts/run_simulation_dgx.sh

The script prints GPU diagnostics, deletes any previous output of the run
directory, regenerates all input files, and launches the solver with unbuffered
output written to ``$JAXALFA_RUNDIR/run.log``.
