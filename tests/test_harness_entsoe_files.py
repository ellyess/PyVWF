"""EntsoeFileSource: file-backed country-level loading (D1 wiring)."""
import pandas as pd
import pytest

from vwf.sources import available_sources, get_source
from vwf.sources.entsoe_files import EntsoeFileSource


@pytest.fixture
def country_layout(tmp_path):
    """Minimal country_level_data layout for a fictional region 'ZZ'."""
    base = tmp_path / "country_level_data"
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
        pd.DataFrame({"capacity_factor": cf}, index=idx).to_csv(path)

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


def test_missing_cluster_column_rejected(tmp_path):
    base = tmp_path / "cl"
    (base / "grid_points" / "zz").mkdir(parents=True)
    pd.DataFrame({"ID": ["g0"], "lon": [5.0], "lat": [52.0], "capacity": [1.0]}).to_csv(
        base / "grid_points" / "zz" / "zz_grid_points.csv", index=False
    )
    src = EntsoeFileSource("ZZ", "train", (2015, 2019), 2023, cl_data_dir=base)
    with pytest.raises(ValueError, match="cluster"):
        src.load_metadata()
