#!/usr/bin/env bash
# Run the sensitivity stages CONCURRENTLY, once the gated E1 result is in.
#
#   bash scripts/pinn/run_remaining.sh [max_jobs]
#
# The stages after E1 are independent of each other and each pins one core:
# torch does not parallelise this workload (many small gathers), so running them
# one at a time leaves seven of eight cores idle. Concurrency is bounded by
# MEMORY, not by cores -- each job holds every region cache at about 2.5 GB --
# so the default of three is set by what 16 GB can hold, not by the core count.
#
# Watch:  tail -f output/pinn/logs/run.log
# Stop:   pkill -f 'scripts/pinn/'
set -u

cd "$(dirname "$0")/../.."
export PYVWF_INPUT=input/combined
export PYTHONPATH=src
PY=/opt/anaconda3/bin/python
LOGS=output/pinn/logs
mkdir -p "$LOGS"
RUN="$LOGS/run.log"
MAX_JOBS="${1:-3}"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN"; }

launch() {
  local name="$1"; shift
  local log="$LOGS/${name}.log"
  (
    t0=$SECONDS
    if "$@" > "$log" 2>&1; then
      say "OK     $name  ($(( (SECONDS - t0) / 60 )) min)  -> $log"
    else
      say "FAILED $name  ($(( (SECONDS - t0) / 60 )) min)  -> $log   [continuing]"
    fi
  ) &
  say "START  $name  (pid $!)"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
}

say "=== sensitivity stages, up to $MAX_JOBS at a time ==="

# Longest first, so the critical path is not a long job started last.
launch 07_e5_mlp           $PY -u scripts/pinn/e1_loro.py \
                               --arms pinn pinn-in-region --hidden 16 \
                               --seeds 0 1 2 --epochs 60 --tag mlp
launch 06_e7_nine_regions  $PY -u scripts/pinn/e1_loro.py \
                               --train-pool DK DE UK US BR NZ CL AR AU-NEM \
                               --regions DK DE UK US BR --arms pinn \
                               --seeds 0 1 2 --epochs 60 --tag nine
launch 04_e4_abstention    $PY -u scripts/pinn/e1_loro.py \
                               --arms pinn-abstain --seeds 0 1 2 \
                               --epochs 60 --tag abstain
launch 05_e3_density       $PY -u scripts/pinn/e1_loro.py \
                               --arms pinn --density --seeds 0 1 2 \
                               --epochs 60 --tag density
launch 03_e2_audit         $PY scripts/pinn/e2_physics_audit.py --epochs 60

wait
say "=== sensitivity stages done ==="

# Figures last: they read whatever completed.
t0=$SECONDS
if $PY scripts/pinn/e1_figures.py --tag primary > "$LOGS/08_figures.log" 2>&1; then
  say "OK     08_figures  ($(( (SECONDS - t0) / 60 )) min)"
else
  say "FAILED 08_figures   -> $LOGS/08_figures.log"
fi
say "=== all done ==="
