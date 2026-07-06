# Scripts Directory

Reusable analysis and processing scripts for the core bias-correction
workflow. See [PIPELINE.md](../PIPELINE.md) for execution order and
dependencies.

## Top-Level Scripts

| Script | Purpose |
|--------|---------|
| `train_all_bias_corrections.py` | Stage-1 orchestrator: train bias corrections across many configurations |
| `evaluate_all_pyvwf_runs.py` | Calculate MAE, RMSE, R² for corrected vs uncorrected capacity factors |
