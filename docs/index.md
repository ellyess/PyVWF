# PyVWF

PyVWF is an open Python framework that turns atmospheric reanalysis (e.g. ERA5)
into **bias-corrected** wind power generation. It re-implements the Virtual Wind
Farm (VWF) methodology behind [Renewables.ninja](https://www.renewables.ninja/),
and — unlike API-only tools — exposes the full *training* workflow for the
correction factors.

Raw reanalysis winds carry systematic, location-dependent biases. PyVWF learns a
per-cluster, per-time-slice linear correction of the wind speed

$$w_{\text{corrected}} = \alpha \cdot w + \beta$$

from observed generation, then converts the corrected wind to power through a
turbine power curve. The factors $\alpha$ (scalar) and $\beta$ (offset) are
yours to inspect, map, and retrain at whatever spatial and temporal resolution
your observations support.

## Where to start

- **New here?** The [project README](https://github.com/ellyess/PyVWF#readme)
  covers installation and a Denmark quickstart.
- **Want to see it run?** `python examples/run_minimal.py` executes the whole
  workflow end-to-end on bundled synthetic data in under a minute — no ERA5
  download and no private turbine data.
- **Looking for a function?** Go to the {doc}`api`.

```{toctree}
:maxdepth: 1
:caption: Guides

DATA_REQUIREMENTS
DATA_PIPELINE
TRAINING_GUIDE
OUTPUT_STRUCTURE
ENTSOE_API_GUIDE
ADDING_AN_OBSERVATION_SOURCE
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Citing PyVWF

Please cite both the software and the method paper.

**The software** (concept DOI — always resolves to the latest release):

> Benmoufok, E. F., Warder, S. C., and Piggott, M. D. *PyVWF: An open Python
> framework for bias-corrected wind power simulation from reanalysis data.*
> Zenodo. [doi:10.5281/zenodo.21236619](https://doi.org/10.5281/zenodo.21236619)

**The method:**

> Benmoufok, E. F., Warder, S. C., Zhu, E., and Piggott, M. D. (2024).
> *Improving wind power modelling through granular spatial and temporal bias
> correction of reanalysis data.* Energy.
> [doi:10.1016/j.energy.2024.133759](https://doi.org/10.1016/j.energy.2024.133759)

Machine-readable metadata for both lives in `CITATION.cff`.
