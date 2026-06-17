# Output Structure

> Extracted from the project README. This reference describes the layout of a
> PyVWF run directory and the files each run produces.

All PyVWF outputs are written to the user-specified output directory (`outdir`)
and organised by run configuration. Each run is fully self-contained.

## Directory layout

```text
outdir/
└── run/
    └── <run_name>/
        ├── plots/
        ├── results/
        │   ├── capacity-factor/
        │   └── wind-speed/
        └── training/
            ├── correction-factors/
            └── simulated-turbines/
```

The `<run_name>` encodes the scenario configuration (e.g. country, correction
mode, surface roughness treatment).

### Plots (`plots/`)

Diagnostic figures summarising model performance:

- `*_full_error_appendix.png`
Overall error metrics across all clusters and time resolutions.

- `*_spatial_focus_error_appendix.png`
Error metrics emphasising spatial structure.

- `*_temporal_focus_error_appendix.png`
Error metrics emphasising temporal variability.

- These figures are intended for appendix or supplementary material.

### Capacity factor results (`results/capacity-factor/`)

CSV time series of simulated and observed capacity factor:

- `<COUNTRY>_<YEAR>_<time_res>_<k>_cor_cf.csv`
Bias-corrected capacity factor for a given time-resolution and cluster count.

- `<COUNTRY>_<YEAR>_unc_cf.csv`
Uncorrected (raw reanalysis-based) capacity factor.

- `<COUNTRY>_<YEAR>_obs_cf.csv`
Observed capacity factor used for validation.

All files share a common time index.
