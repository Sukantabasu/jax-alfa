#!/bin/bash
#SBATCH --job-name=jax_alfa
#SBATCH --output=slurmout/jax-alfa-%j.out
#SBATCH --error=slurmout/jax-alfa-%j.err
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

umask 002
set -eo pipefail

mkdir -p slurmout

# ============================================================
# JAX-ALFA directories
# ============================================================

export JAXALFA_ROOT=/network/rit/lab/basulab/Sukanta/MODELS/JAX-ALFA/JAXALFA0.1
export JAXALFA_RUNDIR=$JAXALFA_ROOT/examples/SBL_GABLS1/runs/200x200x200_LAD_WL_DP

cd "$JAXALFA_ROOT"

# ============================================================
# Conda environment
# ============================================================

export PYTHONNOUSERSITE=1
export PYTHONPATH="$JAXALFA_ROOT:${PYTHONPATH:-}"

source /network/rit/lab/basulab/Sukanta/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/basulab/Sukanta/anaconda3/envs/jax-gpu

# ============================================================
# Diagnostics
# ============================================================

echo "----------------------------------------"
echo "Running JAX-ALFA on DGX"
echo "Node:           $(hostname)"
echo "JAXALFA_ROOT:   $JAXALFA_ROOT"
echo "JAXALFA_RUNDIR: $JAXALFA_RUNDIR"
echo "----------------------------------------"

echo ""
echo "GPU status:"
nvidia-smi

echo ""
echo "JAX devices:"
python - << 'EOF'
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF

echo "----------------------------------------"

# ============================================================
# Remove previous outputs completely
# ============================================================

if [[ -n "$JAXALFA_RUNDIR" && -d "$JAXALFA_RUNDIR/output" ]]; then
    echo "Deleting previous output directory..."
    rm -rf "$JAXALFA_RUNDIR/output"
fi

# ============================================================
# Generate inputs
# ============================================================

echo "Generating input files..."
for f in "$JAXALFA_RUNDIR"/CreateInputs*.py; do
    [ -f "$f" ] && python "$f"
done
for f in "$JAXALFA_RUNDIR"/CreateSurfaceBC*.py; do
    [ -f "$f" ] && python "$f"
done
for f in "$JAXALFA_RUNDIR"/CreateGeoWind*.py; do
    [ -f "$f" ] && python "$f"
done
for f in "$JAXALFA_RUNDIR"/CreateAdvForcing*.py; do
    [ -f "$f" ] && python "$f"
done


# ============================================================
# Run JAX-ALFA
# ============================================================

echo "Starting simulation..."

PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u -m src.Main 2>&1 | tee -a "$JAXALFA_RUNDIR/run.log"

echo "Simulation completed."
