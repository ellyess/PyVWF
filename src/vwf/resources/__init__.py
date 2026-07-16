"""Static reference tables bundled with PyVWF.

These are the **open turbine curve library** (see ``input/README.md``): 69 real
machines plus 7 normalized composites from the NREL/turbine-models
archive (BSD-3-Clause, DOI 10.11578/dc.20210112.1), Gaussian-smoothed to
capacity-factor curves with the published VWF method. Per-column sources and
licenses are recorded in ``power_curves_provenance.csv``. They exist so that a
``pip install pyvwf`` outside a repository checkout can import, run the
example, and be tested on real curve physics.

The library's newest large machines are references and composites rather than
current commercial models, so ``add_models`` matches modern fleets to them by
specific power (rotor loading), not by machine identity. Other power-curve
libraries (renewables.ninja / VWF / thewindpower.net) carry their own licensing
terms and are not redistributed. Point
:attr:`vwf.config.PyVWFPaths.INPUT_ROOT` at a directory containing your own
``power_curves.csv`` and ``models.csv`` and they take precedence over these;
:func:`vwf.config.PyVWFPaths.reference_file` warns whenever it falls back here.
"""
