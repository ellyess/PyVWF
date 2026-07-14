"""Static reference tables bundled with PyVWF.

These are the **synthetic placeholder** power curves and turbine models (see
``input/README.md``): five invented ``Synthetic.*`` models generated from a
simple parametric curve. They exist so that a ``pip install pyvwf`` outside a
repository checkout can still import, run the example, and be tested — not so
that anyone can produce publishable capacity factors from them.

Real power-curve libraries (renewables.ninja / VWF / thewindpower.net) carry
their own licensing terms and are not redistributed. Point
:attr:`vwf.config.PyVWFPaths.INPUT_ROOT` at a directory containing your own
``power_curves.csv`` and ``models.csv`` and they take precedence over these;
:func:`vwf.config.PyVWFPaths.reference_file` warns whenever it falls back here.
"""
