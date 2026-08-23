#!/usr/bin/env bash
# Run the whole physics-informed evaluation programme unattended.
#
#   bash scripts/pinn/run_overnight.sh
#
# Stages run in priority order and are INDEPENDENT: a failure is logged and the
# next stage still runs, so a night that ends early still ends with the most
# important results on disk. Each stage writes its own CSVs under
# output/pinn/, so partial completion is useful rather than wasted.
#
# Everything is appended to output/pinn/logs/. To watch:  tail -f output/pinn/logs/run.log
# To stop:  pkill -f 'scripts/pinn/'
#
# Rough budget, in stage order, on eight cores:
#   00 tests            1 min
#   01 E1 primary     ~4.8 h   <- the gated result; five seeds, as specified
#   02 report           secs
#   03 E2 audit       ~0.3 h
#   04 E4 abstention  ~1.3 h   } three seeds each: sensitivities, not gates,
#   05 E3 density     ~1.3 h   } and the seed spread is already known to be
#   06 E7 nine regions~1.5 h   } about 2e-4 RMSE
#   07 E5 MLP         ~2.5 h
# About 12 hours end to end. Stages are ordered so a night that ends early still
# ends with the gated result and the physics audit on disk.
set -u

cd "$(dirname "$0")/../.."
export PYVWF_INPUT=input/combined
export PYTHONPATH=src
PY=/opt/anaconda3/bin/python
LOGS=output/pinn/logs
mkdir -p "$LOGS"
RUN="$LOGS/run.log"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN"; }

stage() {
  local name="$1"; shift
  local log="$LOGS/${name}.log"
  say "START  $name"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    say "OK     $name  ($(( (SECONDS - t0) / 60 )) min)  -> $log"
  else
    say "FAILED $name  ($(( (SECONDS - t0) / 60 )) min)  -> $log   [continuing]"
  fi
}

say "=== physics-informed ERA5 correction: unattended run ==="
say "commit $(git rev-parse --short HEAD)   curve library: $PYVWF_INPUT"

# 0. Tests first: a broken tree should not consume the whole night.
stage 00_tests            $PY -m pytest tests/test_pinn_physics.py -q

# 1. THE HEADLINE. Pre-specified gates P1, P2, P3 on five regions.
stage 01_e1_primary       $PY -u scripts/pinn/e1_loro.py \
                              --seeds 0 1 2 3 42 --epochs 60 --tag primary
stage 02_e1_report        $PY scripts/pinn/e1_report.py --tag primary

# 2. Is the fitted physics credible, and does addendum 1's density prediction
#    hold? Cheap, and it is what licenses calling any of this physics.
stage 03_e2_audit         $PY scripts/pinn/e2_physics_audit.py --epochs 60

# 3. Abstention outside the physiographic envelope (addendum 4). Directly
#    addresses P1's "degrades nothing by more than 10%" clause.
stage 04_e4_abstention    $PY -u scripts/pinn/e1_loro.py \
                              --arms pinn-abstain --seeds 0 1 2 \
                              --epochs 60 --tag abstain

# 4. Air density as its own arm (addendum 1).
stage 05_e3_density       $PY -u scripts/pinn/e1_loro.py \
                              --arms pinn --density --seeds 0 1 2 \
                              --epochs 60 --tag density

# 5. Nine regions instead of five (addendum 5). Fewer seeds: a sensitivity,
#    not a gate.
stage 06_e7_nine_regions  $PY -u scripts/pinn/e1_loro.py \
                              --train-pool DK DE UK US BR NZ CL AR AU-NEM \
                              --regions DK DE UK US BR --arms pinn \
                              --seeds 0 1 2 --epochs 60 --tag nine

# 6. Capacity sensitivity: linear heads against an MLP (addendum 4, E5).
stage 07_e5_mlp           $PY -u scripts/pinn/e1_loro.py \
                              --arms pinn pinn-in-region --hidden 16 \
                              --seeds 0 1 2 --epochs 60 --tag mlp

# 7. Figures from whatever completed.
stage 08_figures          $PY scripts/pinn/e1_figures.py --tag primary

say "=== done ==="
say "results: output/pinn/e1/*.csv  output/pinn/e2/*.csv  output/pinn/figures/"
