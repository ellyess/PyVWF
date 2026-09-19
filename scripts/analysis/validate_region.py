#!/usr/bin/env python
"""Train, evaluate or transfer one region through the validation harness.

The same command as the ``pyvwf-validate`` console entry, for a checkout
without an install. All logic lives in :mod:`vwf.cli.validate` and
:mod:`vwf.harness.driver`.

Examples:
    python scripts/analysis/validate_region.py train --region configs/regions/dk.toml
    python scripts/analysis/validate_region.py evaluate --region configs/regions/dk.toml \
        --train-run output/validation/DK/train-20260715T120000Z
    python scripts/analysis/validate_region.py transfer --region configs/regions/uk.toml \
        --source-region configs/regions/au_nem.toml \
        --source-run output/validation/AU-NEM/train-20260715T120000Z
"""

import sys

from vwf.cli.validate import main

if __name__ == "__main__":
    sys.exit(main())
