# Input data

## Power curves and turbine models are SYNTHETIC PLACEHOLDERS

The `power_curves.csv` and `models.csv` shipped in this repository are
**synthetic placeholders**. They are generated from a simple parametric model (a
normalised logistic ramp between cut-in and rated wind speed, flat to cut-out,
zero beyond) and exist only so the test suite, the quickstart, and the examples
run out of the box.

They are **not** real power curves, and **not** degraded or rounded real data.
The five `Synthetic.*` models are invented. Do **not** use them for production
resource assessment or for published results.

For real, validated power curves, use the renewables.ninja / Virtual Wind Farm
(VWF) library, which derives from the thewindpower.net turbine database:

- https://github.com/renewables-ninja/vwf
- https://www.renewables.ninja
- https://www.thewindpower.net

Those datasets carry their own licensing terms and are **not redistributed** here.

## Restoring the real curves locally (for your own production runs)

Keep your real files out of git as `input/power_curves.real.csv` and
`input/models.real.csv` (both gitignored), then swap them in when needed:

```bash
cp input/power_curves.real.csv input/power_curves.csv    # use real curves
git checkout input/power_curves.csv input/models.csv     # restore synthetic before committing
```
