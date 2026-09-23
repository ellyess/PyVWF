"""The statistics behind gate K0 of ``docs/findings/method-wake-unit-prereg.md``.

The driver's fits need the ``pinn`` extra, but its statistics do not: the
blocks, the shared-target flag, the weighted slope, the pigeonhole bootstrap
and gate V's two checks are plain numpy, so they are tested here on synthetic
data where the right answer is known. Each check is shown to fire on a known
positive and to return a negative, as the root ``AGENTS.md`` asks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "studies" / "method-wake-unit" / "k0_density_slopes.py"


@pytest.fixture(scope="module")
def k0():
    spec = importlib.util.spec_from_file_location("k0_density_slopes", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocks_link_equal_nonzero_values_and_nothing_else(k0):
    nan = np.nan
    obs = np.array(
        [
            # units 0 and 1 share a value; 2 and 3 share only a zero; 4 alone
            [0.30, 0.30, 0.00, 0.00, 0.10],
            # unit 1 and unit 4 share a value, which joins 0, 1 and 4
            [0.20, 0.25, 0.40, 0.41, 0.25],
            [0.22, nan, 0.40, 0.42, 0.50],
        ]
    )
    block, shared = k0.target_blocks(obs)
    assert block.tolist() == [0, 0, 1, 2, 0]
    # Unit 1 shares in both of its observed months. Every other unit shares in
    # one month of three, which is not more than half; the zero shared by 2
    # and 3 counts towards sharing but links no block.
    assert shared.tolist() == [False, True, False, False, False]


def test_blocks_are_numbered_by_first_unit(k0):
    obs = np.array([[0.1, 0.2, 0.1, 0.2, 0.3]])
    block, _ = k0.target_blocks(obs)
    assert block.tolist() == [0, 1, 0, 1, 2]


def test_weighted_slope_matches_polyfit(k0):
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 5, 200)
    y = 0.7 - 0.03 * x + rng.normal(0, 0.1, 200)
    w = rng.uniform(1, 10, 200)
    expected = np.polyfit(x, y, 1, w=np.sqrt(w))[0]
    assert k0.weighted_slope(x, y, w)[0] == pytest.approx(expected, rel=1e-10)
    stacked = k0.weighted_slope(x, y, np.vstack([w, 2 * w, np.ones_like(w)]))
    assert stacked[0] == pytest.approx(expected, rel=1e-10)
    assert stacked[1] == pytest.approx(expected, rel=1e-10)
    assert stacked[2] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-10)


def _frame(slope: float, seed: int = 0) -> pd.DataFrame:
    """Units in blocks of three over twelve months, residual linear in D plus noise."""
    rng = np.random.default_rng(seed)
    n_units, n_months = 150, 12
    unit = np.arange(n_units)
    block = unit // 3
    d = np.repeat(rng.uniform(0, 3, n_units // 3), 3)
    cap = rng.uniform(1, 5, n_units)
    rows = pd.DataFrame(
        {
            "ID": np.tile(unit.astype(str), n_months),
            "month_idx": np.repeat(np.arange(n_months), n_units),
            "block": np.tile(block, n_months),
            "D": np.tile(d, n_months),
            "capacity": np.tile(cap, n_months),
        }
    )
    month_effect = np.repeat(rng.normal(0, 0.05, n_months), n_units)
    rows["residual"] = slope * rows["D"] + month_effect + rng.normal(0, 0.05, len(rows))
    return rows


def _draws(k0, rows: pd.DataFrame, draws: int = 400):
    rng = np.random.default_rng(0)
    return k0.draw_counts(
        rng, int(rows["block"].max()) + 1, int(rows["month_idx"].max()) + 1, draws
    )


@pytest.mark.parametrize("slope,excludes_zero", [(0.03, True), (0.0, False)])
def test_bootstrap_interval_finds_a_real_slope_and_not_a_null_one(k0, slope, excludes_zero):
    rows = _frame(slope)
    cb, cm = _draws(k0, rows)
    x, y = rows["D"].to_numpy(), rows["residual"].to_numpy()
    b = k0.bootstrap_slopes(
        x,
        y,
        rows["capacity"].to_numpy(),
        rows["block"].to_numpy(),
        rows["month_idx"].to_numpy(),
        cb,
        cm,
    )
    lo, hi = k0.interval(b)
    assert (lo > 0) == excludes_zero
    assert lo <= slope <= hi


def test_bootstrap_counts_are_reproducible_and_sum_to_the_sample_size(k0):
    a = k0.draw_counts(np.random.default_rng(0), 7, 12, 5)
    b = k0.draw_counts(np.random.default_rng(0), 7, 12, 5)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert (a[0].sum(axis=1) == 7).all() and (a[1].sum(axis=1) == 12).all()


def test_gate_v_passes_on_a_sound_frame_while_the_unpermuted_slope_is_found(k0):
    rows = _frame(0.03)
    cb, cm = _draws(k0, rows)
    v = k0._validation(rows, cb, cm)
    assert v["planted_pass"]
    assert v["planted_diff_lo"] == pytest.approx(k0.PLANTED_SLOPE, abs=1e-9)
    assert v["planted_diff_hi"] == pytest.approx(k0.PLANTED_SLOPE, abs=1e-9)
    # The permuted half must return zero here, even though the true slope is
    # not zero: permuting D across blocks breaks the link.
    assert v["permuted_pass"]
    # And the same check fires when the link is intact: without permutation
    # the interval excludes zero.
    b = k0.bootstrap_slopes(
        rows["D"].to_numpy(),
        rows["residual"].to_numpy(),
        rows["capacity"].to_numpy(),
        rows["block"].to_numpy(),
        rows["month_idx"].to_numpy(),
        cb,
        cm,
    )
    assert not (k0.interval(b)[0] <= 0.0 <= k0.interval(b)[1])


def test_permuted_density_keeps_blocks_together(k0):
    d = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    cap = np.ones(5)
    block = np.array([0, 0, 1, 2, 2])
    p = k0.permuted_density(d, cap, block, seed=1)
    assert p[0] == p[1] and p[3] == p[4]
    assert sorted(set(p)) == [1.0, 2.0, 3.0]


def test_cli_refuses_a_value_the_registration_fixes(k0, monkeypatch):
    monkeypatch.setattr(k0, "main", lambda **kw: None)
    with pytest.raises(SystemExit):
        k0.cli(["--out", "x", "--seeds", "0", "1", "2", "3", "4"])
    with pytest.raises(SystemExit):
        k0.cli(["--out", "x", "--draws", "500"])
    k0.cli(["--out", "x", "--seeds", "0", "1", "2", "3", "42", "--draws", "1000"])
