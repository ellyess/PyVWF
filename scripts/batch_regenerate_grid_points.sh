#!/bin/bash
# Generate year-specific grid points for all country-level countries (2015-2021)

set -e

COUNTRIES=("NL" "FR" "BE" "IT" "ES" "PT" "IE" "NO" "SE")
YEARS=(2015 2016 2017 2018 2019 2020 2021)
RADIUS=50

echo "Generating grid points for years 2015-2021..."
echo "=============================================="

for COUNTRY in "${COUNTRIES[@]}"; do
    echo ""
    echo "Country: $COUNTRY"
    for YEAR in "${YEARS[@]}"; do
        echo "  Generating $YEAR..."
        python scripts/regenerate_grid_points_with_gwpt.py \
            --country "$COUNTRY" \
            --year "$YEAR" \
            --radius "$RADIUS" \
            2>&1 | grep -E "(✓|✗|Found|Total)"
    done
done

echo ""
echo "=============================================="
echo "✓ All grid points generated!"
