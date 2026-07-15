"""Tests for the pluggable ObservationSource layer.

Three things are pinned here:

1. The registry contract: registration, resolution, and the errors raised when a
   country has no source. In particular, ``obs_level="country"`` without a source
   must raise NotImplementedError. That path replaced a branch which used to fail
   deep inside the pipeline with a confusing column error, so the new behaviour is
   pinned rather than left to drift.
2. The European turbine adapter honours the contract, including the kWh to
   capacity-factor conversion it inherited from ``prep_country``.
3. A trivial second adapter can be registered and consumed by the pipeline, with
   no change to any core module. This is the seam the Australia adapter will use.

No test touches the real input CSVs; ``input/`` is not tracked by git.
"""
from calendar import monthrange

import pandas as pd
import pytest

import vwf.data as data
import vwf.sources.european as european
from vwf.data import prep_country, train_set
from vwf.sources import (
    EuropeanTurbineSource,
    InMemoryCountrySource,
    ObservationSource,
    available_sources,
    get_source,
    register,
    resolve,
)
from vwf.sources import registry as registry_mod

CAPACITY_KW = 2000.0
TARGET_CF = 0.5


@pytest.fixture
def isolated_registry():
    """Snapshot the registry so tests can register throwaway adapters."""
    snapshot = dict(registry_mod._REGISTRY)
    yield registry_mod
    registry_mod._REGISTRY.clear()
    registry_mod._REGISTRY.update(snapshot)


def _turbine_metadata() -> pd.DataFrame:
    """Metadata shaped as ``add_models`` would leave it."""
    return pd.DataFrame(
        {
            "ID": ["t1", "t2"],
            "type": ["onshore", "offshore"],
            "capacity": [CAPACITY_KW, CAPACITY_KW],
            "diameter": [90.0, 90.0],
            "height": [100.0, 100.0],
            "lon": [8.2, 8.8],
            "lat": [55.2, 55.8],
            "model": ["GE.1.5sle", "GE.1.5sle"],
        }
    )


def _wide_generation(year: int = 2015) -> pd.DataFrame:
    """Raw monthly generation in kWh that corresponds to exactly TARGET_CF."""
    cols: dict[str, list] = {"ID": ["t1", "t2"], "year": [year, year]}
    for month in range(1, 13):
        hours = monthrange(year, month)[1] * 24
        kwh = hours * CAPACITY_KW * TARGET_CF
        cols[str(month)] = [kwh, kwh]
    return pd.DataFrame(cols)


@pytest.fixture
def stub_european_loaders(monkeypatch):
    """Point the European adapter at synthetic CSV-shaped data."""
    calls: dict[str, tuple] = {}

    def fake_metadata(country: str) -> pd.DataFrame:
        calls["metadata"] = (country,)
        return _turbine_metadata()

    def fake_observations(country: str, year_start: int, year_end: int) -> pd.DataFrame:
        calls["observations"] = (country, year_start, year_end)
        return _wide_generation()

    monkeypatch.setattr(european, "load_turbine_metadata", fake_metadata)
    monkeypatch.setattr(european, "load_turbine_observations", fake_observations)
    # add_models is resolved lazily from vwf.data inside load_metadata.
    monkeypatch.setattr(data, "add_models", lambda df: df)
    return calls


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_builtin_sources_are_registered():
    assert "european-turbine" in available_sources()
    assert "in-memory-country" in available_sources()


def test_resolve_turbine_country_is_case_insensitive():
    source = resolve("dk", "turbine")
    assert isinstance(source, EuropeanTurbineSource)
    assert source.country == "DK"
    assert source.obs_level == "turbine"


@pytest.mark.parametrize("country", ["DK", "DE", "UK"])
def test_every_declared_country_resolves(country):
    assert isinstance(resolve(country, "turbine"), EuropeanTurbineSource)


def test_resolve_unknown_turbine_country_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported country=XX"):
        resolve("XX", "turbine")


def test_get_source_by_name_and_unknown_name():
    source = get_source("in-memory-country", pd.DataFrame(), pd.DataFrame())
    assert isinstance(source, InMemoryCountrySource)
    with pytest.raises(KeyError, match="Unknown observation source"):
        get_source("no-such-source")


def test_register_rejects_bad_adapters(isolated_registry):
    with pytest.raises(TypeError):
        register(object)  # type: ignore[arg-type]

    class NoName(ObservationSource):
        obs_level = "turbine"

        def load_metadata(self):
            return pd.DataFrame()

        def load_observations(self, year_start=None, year_end=None):
            return pd.DataFrame()

    NoName.name = ""
    with pytest.raises(ValueError, match="non-empty 'name'"):
        register(NoName)

    class BadLevel(ObservationSource):
        name = "bad-level"
        obs_level = "regional"

        def load_metadata(self):
            return pd.DataFrame()

        def load_observations(self, year_start=None, year_end=None):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="must be 'turbine' or 'country'"):
        register(BadLevel)


def test_register_rejects_duplicate_name(isolated_registry):
    class Duplicate(ObservationSource):
        name = "european-turbine"
        obs_level = "turbine"

        def load_metadata(self):
            return pd.DataFrame()

        def load_observations(self, year_start=None, year_end=None):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="already registered"):
        register(Duplicate)


# ---------------------------------------------------------------------------
# The replaced dead branch: country-level with no source
# ---------------------------------------------------------------------------

def test_resolve_country_level_without_source_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="No country-level observation source"):
        resolve("NL", "country")


def test_prep_country_without_source_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="load_country_data"):
        prep_country("NL", obs_level="country")


def test_train_set_country_level_without_source_raises_not_implemented():
    """Pins the branch that used to fail with a confusing 'output_kwh' column error.

    This must raise before any ERA5 or power-curve file is touched.
    """
    with pytest.raises(NotImplementedError, match="No country-level observation source"):
        train_set("NL", True, obs_level="country")


# ---------------------------------------------------------------------------
# The European turbine adapter satisfies the contract
# ---------------------------------------------------------------------------

def test_european_source_rejects_unsupported_country():
    with pytest.raises(ValueError, match="supports"):
        EuropeanTurbineSource("FR")


def test_european_metadata_satisfies_contract(stub_european_loaders):
    meta = resolve("DK", "turbine").load_metadata()
    required = {"ID", "lon", "lat", "height", "capacity", "model", "type"}
    assert required.issubset(meta.columns)
    assert (meta["height"] > 1.0).all()


def test_european_metadata_is_not_shared_between_callers(stub_european_loaders):
    source = EuropeanTurbineSource("DK")
    first = source.load_metadata()
    first.loc[0, "capacity"] = -1.0
    assert source.load_metadata().loc[0, "capacity"] == CAPACITY_KW


def test_european_observations_convert_kwh_to_capacity_factor(stub_european_loaders):
    obs = EuropeanTurbineSource("DK").load_observations()

    assert list(obs.columns) == ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    assert "capacity" not in obs.columns

    month_cols = [f"obs_{m}" for m in range(1, 13)]
    assert obs[month_cols].to_numpy() == pytest.approx(TARGET_CF)


@pytest.mark.parametrize(
    ("country", "expected"),
    [("DK", (2015, 2019)), ("DE", (2015, 2018)), ("UK", (2015, 2018))],
)
def test_default_train_years_preserved(country, expected, stub_european_loaders):
    EuropeanTurbineSource(country).load_observations()
    assert stub_european_loaders["observations"] == (country, *expected)


def test_explicit_year_overrides_default_window(stub_european_loaders):
    EuropeanTurbineSource("DK").load_observations(2021, 2021)
    assert stub_european_loaders["observations"] == ("DK", 2021, 2021)


def test_prep_country_turbine_passes_year_test_through(stub_european_loaders):
    obs, meta = prep_country("DK", 2021, obs_level="turbine")
    assert stub_european_loaders["observations"] == ("DK", 2021, 2021)
    assert not obs.empty
    assert not meta.empty


# ---------------------------------------------------------------------------
# InMemoryCountrySource
# ---------------------------------------------------------------------------

def _country_observations(periods: int = 48) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "generation_mw": 400.0,
            "capacity_mw": 1000.0,
            "capacity_factor": 0.4,
        },
        index=index,
    )


def _grid_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["grid_0", "grid_1"],
            "lat": [55.2, 55.8],
            "lon": [8.2, 8.8],
            "height": [100.0, 100.0],
            "model": ["GE.1.5sle", "GE.1.5sle"],
            "capacity": [50_000.0, 50_000.0],
            "type": ["onshore", "onshore"],
            "cluster": [0, 1],
        }
    )


def test_in_memory_source_declares_country_level():
    source = InMemoryCountrySource(_grid_points(), _country_observations())
    assert source.obs_level == "country"
    assert source.countries == ()


def test_in_memory_source_ignores_year_arguments():
    source = InMemoryCountrySource(_grid_points(), _country_observations())
    unscoped = source.load_observations()
    scoped = source.load_observations(1999, 1999)
    pd.testing.assert_frame_equal(unscoped, scoped)


def test_in_memory_source_isolates_caller_mutations():
    grid, obs = _grid_points(), _country_observations()
    source = InMemoryCountrySource(grid, obs)

    grid.loc[0, "capacity"] = -1.0
    obs.iloc[0, obs.columns.get_loc("capacity_factor")] = 99.0

    assert source.load_metadata().loc[0, "capacity"] == 50_000.0
    assert source.load_observations()["capacity_factor"].iloc[0] == 0.4


# ---------------------------------------------------------------------------
# A second adapter plugs in without touching the core
# ---------------------------------------------------------------------------

def test_custom_country_source_resolves_and_feeds_prep_country(isolated_registry):
    """Register a brand new region and read it back through the core dispatch."""

    @register
    class AtlantisSource(ObservationSource):
        name = "atlantis"
        obs_level = "country"
        countries = ("ZZ",)

        def __init__(self, country: str) -> None:
            self.country = country.upper()

        def load_metadata(self) -> pd.DataFrame:
            return _grid_points()

        def load_observations(self, year_start=None, year_end=None) -> pd.DataFrame:
            return _country_observations()

    assert isinstance(resolve("ZZ", "country"), AtlantisSource)

    obs, meta = prep_country("ZZ", obs_level="country")
    assert "capacity_factor" in obs.columns
    assert list(meta["ID"]) == ["grid_0", "grid_1"]


def test_custom_source_drives_the_country_level_pipeline(
    isolated_registry, monkeypatch, reanalysis, power_curve
):
    """A fake adapter reaches gen_cf through train_set, with no core edits.

    ERA5 and the power-curve CSV are stubbed because neither is tracked in git.
    Everything between the source and gen_cf is the real pipeline.
    """

    @register
    class AtlantisSource(ObservationSource):
        name = "atlantis"
        obs_level = "country"
        countries = ("ZZ",)

        def __init__(self, country: str) -> None:
            self.country = country.upper()

        def load_metadata(self) -> pd.DataFrame:
            return _grid_points()

        def load_observations(self, year_start=None, year_end=None) -> pd.DataFrame:
            return _country_observations()

    monkeypatch.setattr(
        data, "prep_era5", lambda country, train, calc_z0, **kwargs: reanalysis
    )
    monkeypatch.setattr(data, "load_power_curves", lambda: power_curve)

    gen_cf, turb_info, _, _ = train_set("ZZ", True, obs_level="country")

    assert not gen_cf.empty
    assert {"year", "month", "ID", "sim", "obs", "season", "fixed"}.issubset(gen_cf.columns)
    # The observed capacity factor is the source's, resampled to a monthly mean.
    assert gen_cf["obs"].unique() == pytest.approx([0.4])
    assert set(gen_cf["ID"]) == {"grid_0", "grid_1"}
    assert set(turb_info["cluster"]) == {0, 1}
