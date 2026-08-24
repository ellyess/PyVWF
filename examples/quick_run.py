"""Quick PyVWF demo: train + simulate for a single country/year.

This is a thin wrapper that delegates to the ``pyvwf-train`` console entry point
(installed by ``pip install -e .`` from a checkout). Both invocations are
equivalent:

    python examples/quick_run.py --outdir outputs/demo_DK_2020 --country DK --year-test 2020
    pyvwf-train                --outdir outputs/demo_DK_2020 --country DK --year-test 2020
"""

from vwf.cli.train import main

if __name__ == "__main__":
    main()
