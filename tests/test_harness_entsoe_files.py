"""EntsoeFileSource: file-backed country-level loading (D1 wiring)."""
import pandas as pd
import pytest

from vwf.sources import available_sources, get_source
from vwf.sources.entsoe_files import EntsoeFileSource


@pytest.fixture
def country_layout(tmp_path):
    """Minimal observations/country layout for a fictional region 'ZZ'."""
    base = tmp_path / "observations/country"
    (base / "grid_points" / "zz").mkdir(parents=True)
    (base / "observations" / "zz").mkdir(parents=True)

    pd.DataFrame(
        {
            "ID": ["g0", "g1"],
            "lon": [5.0, 6.0],
            "lat": [52.0, 53.0],
            "height": [100.0, 100.0],
            "capacity": [3000.0, 3000.0],
            "model": ["M", "M"],
            "type": ["onshore", "onshore"],
            "cluster": [0, 1],
        }
    ).to_csv(base / "grid_points" / "zz" / "zz_grid_points.csv", index=False)

    def _obs(path, cf):
        idx = pd.date_range("2015-01-01", periods=4, freq="15min", tz="UTC")
        # First value is the one the assertions read; the rest carry the series
        # over the plausibility gate in vwf.loaders.country_obs_checks, which
        # runs on every load and would otherwise warn about a flat series.
        pd.DataFrame({"capacity_factor": [cf, 0.05, 0.9, 0.3]}, index=idx).to_csv(path)

    _obs(base / "observations" / "zz" / "zz_train_2015_2019.csv", 0.11)
    _obs(base / "observations" / "zz" / "zz_test_2023.csv", 0.22)
    return base


def test_registered():
    assert "entsoe-country" in available_sources()
    # Not auto-resolvable: constructed explicitly, so get_source needs its args.
    src = get_source(
        "entsoe-country", "ZZ", "train", (2015, 2019), 2023
    )
    assert isinstance(src, EntsoeFileSource)


def test_train_and_test_read_different_files(country_layout):
    train = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=country_layout)
    test = EntsoeFileSource("ZZ", "test", (2015, 2019), 2023, cl_data_dir=country_layout)

    gp = train.load_metadata()
    assert list(gp["cluster"]) == [0, 1]
    assert gp["ID"].tolist() == ["g0", "g1"]

    assert train.load_observations()["capacity_factor"].iloc[0] == pytest.approx(0.11)
    assert test.load_observations()["capacity_factor"].iloc[0] == pytest.approx(0.22)


def test_bad_split_rejected():
    with pytest.raises(ValueError, match="split must be"):
        EntsoeFileSource("ZZ", "validation", (2015, 2019), 2023)


def test_missing_files_raise(tmp_path):
    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="grid points"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="observations"):
        src.load_observations()


def test_aggregated_suffix_is_accepted(country_layout):
    """Zonal regions (NO, SE) have no single national series; their national
    file is the sum over bidding zones, written with an ``_aggregated`` suffix.
    Both spellings must resolve, or those regions cannot be trained at all."""
    obs_dir = country_layout / "observations" / "zz"
    plain = obs_dir / "zz_train_2015_2019.csv"
    plain.rename(obs_dir / "zz_train_2015_2019_aggregated.csv")

    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=country_layout)
    assert src.load_observations()["capacity_factor"].iloc[0] == pytest.approx(0.11)


def test_plain_name_wins_over_aggregated(country_layout):
    """When both spellings exist the national file is preferred, so adding an
    aggregate never silently displaces a real national series."""
    obs_dir = country_layout / "observations" / "zz"
    idx = pd.date_range("2015-01-01", periods=4, freq="15min", tz="UTC")
    pd.DataFrame({"capacity_factor": [0.99, 0.05, 0.9, 0.3]}, index=idx).to_csv(
        obs_dir / "zz_train_2015_2019_aggregated.csv"
    )

    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=country_layout)
    assert src.load_observations()["capacity_factor"].iloc[0] == pytest.approx(0.11)


def test_missing_observations_names_every_candidate(tmp_path):
    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="_aggregated"):
        src.load_observations()


def test_implausible_series_warns_on_load(country_layout):
    """The gate has to fire where the data enters the model, not only in an
    audit script someone has to remember to run."""
    obs_dir = country_layout / "observations" / "zz"
    idx = pd.date_range("2015-01-01", periods=8, freq="h", tz="UTC")
    # NL's signature: every value scaled down, so the fleet never approaches
    # its own peak. In-sample the correction absorbs it and looks fine.
    pd.DataFrame({"capacity_factor": [0.05] * 7 + [0.09]}, index=idx).to_csv(
        obs_dir / "zz_test_2023.csv"
    )

    src = EntsoeFileSource("ZZ", "test", (2015, 2019), 2023, cl_data_dir=country_layout)
    with pytest.warns(UserWarning, match="never reaches"):
        src.load_observations()


def test_missing_cluster_column_rejected(tmp_path):
    base = tmp_path / "cl"
    (base / "grid_points" / "zz").mkdir(parents=True)
    pd.DataFrame({"ID": ["g0"], "lon": [5.0], "lat": [52.0], "capacity": [1.0]}).to_csv(
        base / "grid_points" / "zz" / "zz_grid_points.csv", index=False
    )
    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=base)
    with pytest.raises(ValueError, match="cluster"):
        src.load_metadata()
