# Thesis Chapter Figures

This directory holds figure-generation scripts for the author's PhD thesis
(Imperial College London, Earth Science & Engineering). They are **not** part
of the reusable PyVWF pipeline and are unlikely to run cleanly outside the
author's research environment - paths point at `output/runs/...` artefacts
produced by the full Stage 1/2/3 pipeline against private datasets (ERA5,
turbine fleet data, etc.).

For the maintained, reusable pipeline see [`../scripts/`](../scripts/) and
[`../PIPELINE.md`](../PIPELINE.md).

## Contents

| Script | Thesis chapter | Topic |
|--------|---------------|-------|
| `generate_ch3_plots.py`      | 3 | PyVWF framework (DK onshore turbine-level results) |
| `generate_ch4_grid_plots.py` | 4 | Grid interpolation of corrections |
| `generate_ch5_ml_plots.py`   | 5 | ML correction models |

Each script applies the shared plot style from `vwf.viz.style` and palette
from `vwf.viz.palettes`. Run from the repo root, e.g.:

```bash
python thesis_figures/generate_ch3_plots.py
```
