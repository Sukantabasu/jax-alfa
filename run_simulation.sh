#!/bin/bash

# ============================================================
# Set the run directory below before running
# ============================================================

export JAXALFA_RUNDIR=/data/Sukanta/MODELS/JAX-ALFA/JAXALFA0.1/examples/DC_Wangara/runs/80x80x80_LASDD_SM_SP

# ============================================================
# Run simulation (do not edit below this line)
# ============================================================

cd "$(dirname "$0")"
rm -rf $JAXALFA_RUNDIR/output
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
python -m src.Main 2>&1 | tee $JAXALFA_RUNDIR/run.log
