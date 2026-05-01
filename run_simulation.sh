#!/bin/bash

# ============================================================
# Set the run directory below before running
# ============================================================

export JAXALFA_RUNDIR=/path/to/examples/CBL_N91/runs/128x128x128

# ============================================================
# Run simulation (do not edit below this line)
# ============================================================

cd "$(dirname "$0")"
rm -rf $JAXALFA_RUNDIR/output
python $JAXALFA_RUNDIR/CreateInputs*.py
python -m src.Main 2>&1 | tee $JAXALFA_RUNDIR/run.log
