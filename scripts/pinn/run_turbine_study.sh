#!/usr/bin/env bash
# The turbine-only terrain study, as its registration records it.
#
#   nohup caffeinate -i bash scripts/pinn/run_turbine_study.sh > <log> 2>&1 &
#
# About twelve hours for the primary run, plus about ninety minutes for the
# German centroid comparison it queues after it. Every stage must succeed
# before the next starts: the first attempt on 2026-09-20 ran the study on
# centroid GWA ratios under a MaStR cache because a failed prerequisite did
# not stop it.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
[ -z "$(git status --porcelain)" ] || { echo "tree dirty, stop"; exit 1; }
echo "HEAD $(git rev-parse --short HEAD) start $(date +%FT%T)"

REG=docs/findings/method-physics-informed-turbine-prereg.md
LOCO=output/pinn_loco_2026-09-16/cache
MASTR=output/pinn_turbine_2026-09-18/cache_mastr
GWA=output/pinn_turbine_2026-09-18/gwa
GWA_MASTR=output/pinn_turbine_2026-09-18/gwa_mastr
STUDY=output/pinn_turbine_2026-09-18/study
OVERRIDE=output/pinn_turbine_2026-09-18/mastr/de_fleet_override.csv
PY=/opt/miniconda3/envs/pyvwf/bin/python

stage() { echo; echo "=== $* $(date +%T)"; }

# The ratios: every region at its cache's own locations, then DE again at the
# rebuilt ones. A ratio computed elsewhere would not describe the unit.
if [ ! -f "$GWA/DK_train.csv" ]; then
  stage "gwa_ratio on the leave-one-country-out caches"
  PYTHONPATH=src $PY -u scripts/pinn/gwa_ratio.py --cache "$LOCO" --gwa input/raw/gwa4 --out "$GWA"
fi
if [ ! -f "$GWA_MASTR/DE_train.csv" ]; then
  stage "gwa_ratio on the rebuilt German caches"
  PYTHONPATH=src $PY -u scripts/pinn/gwa_ratio.py --regions DE --splits train test \
    --cache "$MASTR" --gwa input/raw/gwa4 --out "$GWA_MASTR"
fi

CONFIGS=(--config DK=configs/regions/scorecard/dk_k100.toml --config DE=configs/regions/scorecard/de_k100.toml
  --config UK=configs/regions/scorecard/uk_k50.toml --config US=configs/regions/scorecard/us_k250.toml
  --config BR=configs/regions/scorecard/br_k60.toml --config AR=configs/regions/scorecard/ar_k10.toml
  --config AU-NEM=configs/regions/scorecard/au_nem_k45.toml --config CL=configs/regions/scorecard/cl_k10.toml
  --config NZ=configs/regions/scorecard/nz_k7.toml --config FR=configs/regions/scorecard/fr_country.toml
  --config BE=configs/regions/scorecard/be_country.toml --config ES=configs/regions/scorecard/es_country.toml
  --config IE=configs/regions/scorecard/ie_country.toml --config IT=configs/regions/scorecard/it_country.toml
  --config NO=configs/regions/scorecard/no_country.toml --config SE=configs/regions/scorecard/se_country.toml
  --config PT=configs/regions/scorecard/pt_country.toml --config NL=configs/regions/pinn_loco/nl_country.toml)

stage "turbine_loro primary (DE on the rebuilt caches and their ratios)"
PYVWF_INPUT=input/combined PYTHONPATH=src $PY -u scripts/pinn/turbine_loro.py \
  --tag primary --cache "$LOCO" --cache-override DE="$MASTR" --gwa-override DE="$GWA_MASTR" \
  --score-subset DE="$OVERRIDE" --gwa "$GWA" --out "$STUDY" "${CONFIGS[@]}" --registration "$REG"

stage "turbine_loro de-centroid (the same DE fold on the centroid caches and ratios)"
PYVWF_INPUT=input/combined PYTHONPATH=src $PY -u scripts/pinn/turbine_loro.py \
  --tag de-centroid --cache "$LOCO" --folds DE --score-subset DE="$OVERRIDE" --no-secondary \
  --gwa "$GWA" --out "$STUDY" "${CONFIGS[@]}" --registration "$REG"

echo; echo "ALL DONE $(date +%FT%T); tree: $(git status --porcelain | wc -l) changes"
