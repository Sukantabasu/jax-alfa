#!/bin/bash
#SBATCH --job-name=jax_alfa
#SBATCH --output=slurmout/jax-alfa-%j.out
#SBATCH --error=slurmout/jax-alfa-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
# ============================================================
# run_simulation_dgx.sh — run one JAX-ALFA case on a SLURM GPU cluster.
#
# Edit the four items marked EDIT, then submit from the repository root:
#     mkdir -p slurmout && sbatch scripts/run_simulation_dgx.sh
# Adjust the #SBATCH lines above (time, memory, GPUs, and if needed
# --account / --partition / --qos) to your cluster.
# Previous output/ and run.log of the run directory are deleted first.
# ============================================================

umask 002
set -eo pipefail

# Avoid BLAS/OpenMP thread and workspace over-allocation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ------------------------------------------------------------
# EDIT 1-2: repository and run directory
# ------------------------------------------------------------
export JAXALFA_ROOT=/path/to/jax-alfa
export JAXALFA_RUNDIR=$JAXALFA_ROOT/examples/SBL_GABLS1/runs/128x128x128_LASDD_SM_DP

# ------------------------------------------------------------
# EDIT 3-4: conda installation and JAX-enabled environment
# ------------------------------------------------------------
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate /path/to/anaconda3/envs/jax-gpu

# ------------------------------------------------------------
# No need to edit below this line
# ------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JAXALFA_ROOT:${PYTHONPATH:-}"
cd "$JAXALFA_ROOT"

echo "----------------------------------------"
echo "Node:           $(hostname)"
echo "JAXALFA_ROOT:   $JAXALFA_ROOT"
echo "JAXALFA_RUNDIR: $JAXALFA_RUNDIR"
echo "----------------------------------------"
nvidia-smi
python -c "import jax; print('JAX', jax.__version__, jax.devices())"

rm -rf "$JAXALFA_RUNDIR/output"
rm -f  "$JAXALFA_RUNDIR/run.log"

echo "Generating input files..."
for f in "$JAXALFA_RUNDIR"/Create*.py; do
    [ -f "$f" ] && python "$f"
done

echo "Starting simulation..."
PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u -m src.Main 2>&1 | tee "$JAXALFA_RUNDIR/run.log"
echo "Simulation completed."
