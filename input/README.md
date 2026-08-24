# Input data

## Power curves and turbine models: the bundled open library

The `power_curves.csv` and `models.csv` shipped in this repository are the
**open turbine curve library**: 69 real machines plus 7 normalized composites
from the [NREL/turbine-models](https://github.com/NREL/turbine-models)
(now hosted as NatLabRockies/turbine-models)
archive (BSD-3-Clause, [DOI 10.11578/dc.20210112.1](https://doi.org/10.11578/dc.20210112.1)),
Gaussian-smoothed to capacity-factor curves with the published VWF method
(Staffell and Pfenninger, 2016). Per-column sources, licenses, and the smoothing
configuration are recorded in `power_curves_provenance.csv`.

These are real measured and reference curves, so the test suite, the
quickstart, and the examples run on real curve physics out of the box. Two
limitations to know before using them for results:

- The library's large machines are mostly research references (NREL, DTU, IEA,
  BAR) and market-average composites rather than current commercial models, so
  `add_models` matches a modern fleet to them by **specific power** (rotor
  loading), not by actual machine identity. For the Danish and German fleets
  this places 94 to 99% of capacity within 20% of the turbine's true specific
  power.
- The 7 normalized composite profiles (flagged `is_normalized_composite` in
  the provenance file) are dimensionless: they appear in
  the curve file (useful as generic defaults) but deliberately not in
  `models.csv`, so the matcher never mistakes them for real machines.

For manufacturer-specific validated curves, use the renewables.ninja / Virtual
Wind Farm (VWF) library, which derives from the thewindpower.net turbine
database:

- https://github.com/renewables-ninja/vwf
- https://www.renewables.ninja
- https://www.thewindpower.net

Those datasets carry their own licensing terms and are **not redistributed** here.

## Using your own curve library (for production runs)

Keep licensed files out of git as `input/reference/power_curves.real.csv` and
`input/reference/models.real.csv` (both gitignored), then swap them in when needed:

```bash
cp input/reference/power_curves.real.csv input/reference/power_curves.csv    # use your own curves
git checkout input/reference/power_curves.csv input/reference/models.csv     # restore the open library before committing
```

## Folder layout

`input/` is organised by pipeline stage:

```
input/
  raw/<source>/                  upstream downloads (provenance): aemo, cammesa, cen, eia, emi, ons, repd
  era5/                          reanalysis, per region
  observations/{turbine,country}/  processed capacity factors the adapters read
  reference/                     static shared lookups: power_curves.csv, models.csv, shapes/, terrain/, gwpt/
  combined/                      alternate run root: same data via symlink + the merged (open+licensed) curve library
```

Point `PYVWF_INPUT` at `input/combined` to run with the fuller curve library.
