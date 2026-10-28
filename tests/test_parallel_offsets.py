"""The parallel offset fit: identical when it runs, loud when it cannot (#80).

``PYVWF_OFFSET_WORKERS`` above 1 fans the row-wise offset fit across
``dask.distributed`` workers. Until 2026-10-16 any failure, from a missing
``distributed`` to a worker crash, silently re-ran the fit sequentially, and
the only test of the parallel path was a real-data pin CI never runs. These
cases run on the synthetic Denmark fixture.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import pyvwf.correction as correction
from pyvwf.data import cluster_train_set, train_set
from pyvwf.harness import corrections
from support import DK_BBOX


@pytest.fixture
def rows(synthetic_dk):
    """The valid rows, clusters, reanalysis and curves of one synthetic fit."""
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore", era5_dir=synthetic_dk["era5"], bbox=DK_BBOX
    )
    bias_df, clus_info = cluster_train_set(gen_cf, "fixed", 2, turb_info)
    valid = bias_df[bias_df["obs"].notna() & (bias_df["obs"] > 0)].copy()
    return valid, clus_info, reanalysis, power_curves


def sequential(valid, clus_info, reanalysis, power_curves):
    return valid.apply(correction.find_offset, args=(clus_info, reanalysis, power_curves), axis=1)


@pytest.fixture
def two_workers(monkeypatch):
    monkeypatch.setenv("PYVWF_OFFSET_WORKERS", "2")
    monkeypatch.setattr(corrections, "_PARALLEL_OFFSET_MIN_ROWS", 0)


def test_the_parallel_fit_is_bit_identical_to_the_sequential_one(rows, two_workers):
    pytest.importorskip("distributed")
    valid, clus_info, reanalysis, power_curves = rows
    got = corrections._fit_offsets(valid, clus_info, reanalysis, power_curves, None)
    want = sequential(valid, clus_info, reanalysis, power_curves)
    pd.testing.assert_series_equal(
        got.sort_index(), want.sort_index(), check_names=False, check_exact=True
    )


def test_a_missing_distributed_warns_and_fits_sequentially(rows, two_workers, monkeypatch):
    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "dask.distributed", None)
    valid, clus_info, reanalysis, power_curves = rows
    with pytest.warns(RuntimeWarning, match="parallel"):
        got = corrections._fit_offsets(valid, clus_info, reanalysis, power_curves, None)
    pd.testing.assert_series_equal(got, sequential(valid, clus_info, reanalysis, power_curves))


def _fake_distributed(monkeypatch, cluster):
    fake = types.ModuleType("dask.distributed")
    fake.LocalCluster = cluster

    class Client:
        def __init__(self, _cluster):
            pass

        def scatter(self, obj, broadcast=True):
            return obj

        def close(self):
            pass

    fake.Client = Client
    monkeypatch.setitem(sys.modules, "dask.distributed", fake)


def test_a_cluster_that_fails_to_start_warns_and_fits_sequentially(rows, two_workers, monkeypatch):
    def refuse(**_kwargs):
        raise OSError("no ports")

    _fake_distributed(monkeypatch, refuse)
    valid, clus_info, reanalysis, power_curves = rows
    with pytest.warns(RuntimeWarning, match="did not start"):
        got = corrections._fit_offsets(valid, clus_info, reanalysis, power_curves, None)
    pd.testing.assert_series_equal(got, sequential(valid, clus_info, reanalysis, power_curves))


def test_a_failure_on_the_workers_propagates(rows, two_workers, monkeypatch):
    """Known positive for the old behaviour's defect: a crash no longer looks slow."""
    closed = []

    class Cluster:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            closed.append(True)

    _fake_distributed(monkeypatch, Cluster)
    import dask.dataframe as dd

    def crash(*_args, **_kwargs):
        raise MemoryError("worker killed")

    monkeypatch.setattr(dd, "from_pandas", crash)
    valid, clus_info, reanalysis, power_curves = rows
    with pytest.raises(MemoryError, match="worker killed"):
        corrections._fit_offsets(valid, clus_info, reanalysis, power_curves, None)
    assert closed == [True]


def test_below_the_row_threshold_it_is_sequential_and_silent(rows, monkeypatch, recwarn):
    monkeypatch.setenv("PYVWF_OFFSET_WORKERS", "2")
    monkeypatch.setitem(sys.modules, "dask.distributed", None)
    valid, clus_info, reanalysis, power_curves = rows
    assert len(valid) < corrections._PARALLEL_OFFSET_MIN_ROWS
    corrections._fit_offsets(valid, clus_info, reanalysis, power_curves, None)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
