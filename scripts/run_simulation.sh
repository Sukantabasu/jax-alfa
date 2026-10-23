#!/bin/bash
# ============================================================
# run_simulation.sh — run one JAX-ALFA case on a workstation.
#
# Usage (from anywhere):  bash scripts/run_simulation.sh [run directory]
# The run directory can be given as an argument or by editing JAXALFA_RUNDIR below.
# Previous output/ and run.log of that run directory are deleted first.
# ============================================================

# ------------------------------------------------------------
# Run directory (edit, or pass as the first argument)
# ------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/.." && pwd)
export JAXALFA_RUNDIR=${1:-$REPO/examples/SBL_GABLS1/runs/64x64x64_LASDD_SM_SP}

# ------------------------------------------------------------
# Run simulation (no need to edit below this line)
# ------------------------------------------------------------
set -eo pipefail
JAXALFA_RUNDIR=$(cd "$JAXALFA_RUNDIR" && pwd)
[ -f "$JAXALFA_RUNDIR/Config.py" ] || { echo "No Config.py in $JAXALFA_RUNDIR" >&2; exit 1; }
cd "$REPO"

rm -rf "$JAXALFA_RUNDIR/output"
rm -f  "$JAXALFA_RUNDIR/run.log"

# Generate initial conditions and forcing files (CreateInputs, CreateSurfaceBC,
# CreateGeoWind, CreateAdvForcing, ... whichever the case provides)
for f in "$JAXALFA_RUNDIR"/Create*.py; do
    [ -f "$f" ] && python "$f"
done

PYTHONUNBUFFERED=1 python -u -m src.Main 2>&1 | tee "$JAXALFA_RUNDIR/run.log"
