# PyVWF

PyVWF is an open Python framework that turns atmospheric reanalysis (e.g. ERA5)
into **bias-corrected** wind power generation. It began as a Python
implementation of the Virtual Wind Farm (VWF) model behind
[Renewables.ninja](https://www.renewables.ninja/) and has grown into its own
model, extending it with a granular bias correction validated against observed
generation, and, unlike API-only tools, exposes the full *training* workflow
for the factors.

Raw reanalysis winds carry systematic, location-dependent biases. PyVWF learns a
per-cluster, per-time-slice affine correction of the wind speed

$$w_{\text{corrected}} = \alpha \cdot w + \beta$$

from observed generation, then converts the corrected wind to power through a
turbine power curve. The factors $\alpha$ (scalar) and $\beta$ (offset) are
yours to inspect, map, and retrain at whatever spatial resolution and time
slice your observations support.

## Where to start

- **New here?** The [project README](https://github.com/ellyess/PyVWF#readme)
  says what PyVWF is and what it is for, in two minutes.
- **Installing it?** The {doc}`installation guide <guides/installation>` covers
  the two routes and the six extras, and {doc}`guides/docker` covers the image.
- **Want to see it run?** `python examples/run_minimal.py` executes the whole
  workflow end-to-end in a few seconds on bundled data (synthetic weather and
  observations, open-library power curves), with no ERA5 download and no private
  turbine data.
- **Looking for a function?** Go to the {doc}`api`.
- **Want the validation numbers?** The per-region scorecard, and the research
  records behind it, live in `docs/findings/` in the repository rather than on
  this site. Each is one dated experiment against one held-out test year, and
  several record negative results, so they are kept where they can be read with
  their full context instead of presented here as guidance. Start at
  [findings/scorecard.md](https://github.com/ellyess/PyVWF/blob/main/docs/findings/scorecard.md).

```{toctree}
:maxdepth: 1
:caption: Guides

guides/installation
guides/data-sources
guides/training
guides/output-structure
guides/visualisation
guides/adding-a-region
guides/adding-an-adapter
guides/adding-a-study
guides/your-own-data
guides/docker
```

```{toctree}
:maxdepth: 1
:caption: Region runbooks

runbooks/dk
runbooks/uk
runbooks/de
runbooks/es
runbooks/us
runbooks/br
runbooks/nz
runbooks/cl
runbooks/ar
runbooks/au_nem
runbooks/entsoe
runbooks/tr
```

```{toctree}
:maxdepth: 1
:caption: Design

design/harness
design/limitations
design/roughness-temporal-treatment
design/undefined-roughness-in-complex-terrain
design/agent-guards
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
CONTEXT
publications
```

## Citing PyVWF

Cite both the software and the method paper. Both references, with the concept
DOI that resolves to the latest release, are in the
[project README](https://github.com/ellyess/PyVWF#citation) and, machine
readable, in `CITATION.cff`. The commit behind each published result is in
{doc}`publications`.
