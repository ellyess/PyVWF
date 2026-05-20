"""Integration tests for the quantile-mapping path wired into PyVWF.

These drive the real ``PyVWF`` helper methods (factor fitting + monthly
application) on synthetic data, so the end-to-end wiring is exercised without
needing ERA5 reanalysis or the ENTSO-E API.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.vwf import PyVWF
from vwf.time_utils import add_time_resolution_columns


@pytest.fixture
def qm_model(tmp_path):
    return PyVWF(
        path=str(tmp_path / "run"),
        country="DK",
        correct=True,
        calc_z0=True,
        cluster_mode="all",
        cluster_list=[2],
        time_res_list=["season"],
        obs_level="turbine",
        correction_method="quantile",
        qm_n_quantiles=20,
    )


def test_run_name_marks_quantile(qm_model):
    assert "-qm" in qm_model.directory_path
    assert qm_model.correction_method == "quantile"


def _synthetic_monthly_gen_cf(ids, clusters, years=(2015, 2016, 2017), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for tid, clu in zip(ids, clusters):
        for yr in years:
            for month in range(1, 13):
                obs = float(np.clip(rng.normal(0.28 + 0.04 * clu, 0.07), 0.02, 0.9))
                sim = float(np.clip(obs * 1.4 + 0.05, 0.02, 0.97))
                rows.append({"ID": tid, "year": yr, "month": month, "sim": sim, "obs": obs})
    df = pd.DataFrame(rows)
    return add_time_resolution_columns(df)


def test_fit_and_apply_quantile_end_to_end(qm_model):
    ids = ["t0", "t1", "t2", "t3"]
    clusters = [0, 0, 1, 1]
    clus_info = pd.DataFrame({"ID": ids, "cluster": clusters})

    gen_cf = _synthetic_monthly_gen_cf(ids, clusters)

    # --- fit + save factors (mirrors PyVWF.train quantile branch) ---
    frame = qm_model._fit_and_save_quantile_factors(gen_cf, clus_info, 2, "season")
    factor_path = qm_model._qm_factor_path(2, "season")
    import os
    assert os.path.exists(factor_path)
    assert {"cluster", "season", "p", "model_q", "obs_q"}.issubset(frame.columns)

    # --- build a synthetic uncorrected CF file for a test year ---
    times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(1)
    unc = pd.DataFrame({"time": times})
    for tid, clu in zip(ids, clusters):
        base = 0.28 + 0.04 * clu
        unc[tid] = np.clip(base * 1.4 + 0.05 + rng.normal(0, 0.12, len(times)), 0.02, 0.97)
    unc_path = str(qm_model.directory_path) + "/results/capacity-factor/DK_2020_unc_cf.csv"
    unc.to_csv(unc_path, index=False)

    out_path = str(qm_model.directory_path) + "/results/capacity-factor/DK_2020_season_2_qm_cf.csv"
    wide = qm_model._simulate_quantile_cf(unc_path, clus_info, 2, "season", out_path)

    # Output is monthly (12 rows), one column per turbine plus time.
    assert os.path.exists(out_path)
    assert "time" in wide.columns
    assert len(wide) == 12
    for tid in ids:
        assert tid in wide.columns

    # Correction should pull the simulated monthly distribution toward the
    # observed one: simulated CF was inflated (x1.4), so corrected variance
    # should drop relative to the uncorrected monthly series.
    unc_monthly = unc.set_index("time").resample("ME").mean()
    for tid in ids:
        unc_var = np.var(unc_monthly[tid].to_numpy())
        cor_var = np.var(wide[tid].to_numpy())
        assert cor_var <= unc_var + 1e-9
        assert np.all((wide[tid] >= 0) & (wide[tid] <= 1))
