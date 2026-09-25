"""Region-config loading and validation (docs/design/harness.md §1)."""

from pathlib import Path

import pytest

from pyvwf.harness.regions import RegionSpec, load_region, season_of_month

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "regions"

VALID = """
[region]
code = "ZZ"
name = "Testland"

[observations]
source = "test-source"
obs_level = "turbine"
obs_unit = "farm"
train_years = [2015, 2018]
test_years = [2019]

[era5]
path = "era5/ZZ"
bbox = [0.0, 10.0, 40.0, 50.0]
file_tag = "ZZ"

[correction]
model = "affine-wind"
cluster_list = [5]
time_slices = ["fixed", "season"]

[seasons]
summer = [6, 7, 8]
autumn = [9, 10, 11]
winter = [12, 1, 2]
spring = [3, 4, 5]
"""


def write_config(tmp_path, text=VALID, **replacements):
    """Write a config derived from VALID with line-level replacements."""
    for old, new in replacements.items():
        assert old in text, f"replacement target {old!r} not in template"
        text = text.replace(old, new)
    path = tmp_path / "region.toml"
    path.write_text(text)
    return path


def test_valid_config_loads(tmp_path):
    spec = load_region(write_config(tmp_path))
    assert spec.code == "ZZ"
    assert spec.source == "test-source"
    assert spec.obs_level == "turbine"
    assert spec.obs_unit == "farm"
    assert spec.train_years == (2015, 2018)
    assert spec.test_years == (2019,)
    assert spec.bbox == (0.0, 10.0, 40.0, 50.0)
    assert spec.cluster_list == (5,)
    assert spec.time_slices == ("fixed", "season")
    assert spec.seasons["winter"] == (12, 1, 2)
    assert spec.pseudo_replicated_rows is False
    assert spec.location_resolution is None
    assert spec.time_convention == "utc-monthly-bins"


def test_season_of_month_covers_every_month(tmp_path):
    spec = load_region(write_config(tmp_path))
    mapping = season_of_month(spec)
    assert sorted(mapping) == list(range(1, 13))
    assert mapping[1] == "winter" and mapping[7] == "summer"


@pytest.mark.parametrize(
    "old, new, match",
    [
        ('obs_level = "turbine"', 'obs_level = "farm"', "obs_level"),
        ('obs_unit = "farm"', 'obs_unit = "station"', "obs_unit"),
        ("train_years = [2015, 2018]", "train_years = [2018, 2015]", "start <= end"),
        ("train_years = [2015, 2018]", "train_years = [2015]", "pair"),
        ("test_years = [2019]", "test_years = [2016]", "inside the training window"),
        ("bbox = [0.0, 10.0, 40.0, 50.0]", "bbox = [10.0, 0.0, 40.0, 50.0]", "longitudes"),
        ("bbox = [0.0, 10.0, 40.0, 50.0]", "bbox = [0.0, 200.0, 40.0, 50.0]", "longitudes"),
        ("bbox = [0.0, 10.0, 40.0, 50.0]", "bbox = [0.0, 10.0, 50.0, 40.0]", "latitudes"),
        ("bbox = [0.0, 10.0, 40.0, 50.0]", "bbox = [0.0, 10.0, 40.0]", "bbox"),
        ('time_slices = ["fixed", "season"]', 'time_slices = ["weekly"]', "unknown time_slices"),
        ("cluster_list = [5]", "cluster_list = [0]", ">= 1"),
        ("cluster_list = [5]", "cluster_list = []", "non-empty"),
        ("summer = [6, 7, 8]", "summer = [7, 8]", "partition"),  # month 6 missing
        ("spring = [3, 4, 5]", "spring = [3, 4, 5, 9]", "partition"),  # month 9 twice
    ],
)
def test_invalid_configs_fail_with_field_in_message(tmp_path, old, new, match):
    path = write_config(tmp_path, **{old: new})
    with pytest.raises(ValueError, match=match):
        load_region(path)


@pytest.mark.parametrize(
    "old, new, match",
    [
        # A misspelt optional key used to be ignored and its default applied.
        ('file_tag = "ZZ"', 'file_tag = "ZZ"\nroughnes = "stored"', r"roughnes.*\[era5\]"),
        (
            "cluster_list = [5]",
            "cluster_list = [5]\nmin_cluster = 3",
            r"min_cluster.*\[correction\]",
        ),
        ('name = "Testland"', 'name = "Testland"\n\n[provenance]\nnote = "x"', r"\[provenance\]"),
    ],
)
def test_unknown_keys_and_sections_fail(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        load_region(write_config(tmp_path, **{old: new}))


def test_season_names_are_free(tmp_path):
    text = VALID.replace("summer", "wet").replace("winter", "dry")
    spec = load_region(write_config(tmp_path, text=text))
    assert set(spec.seasons) == {"wet", "autumn", "dry", "spring"}


def test_every_committed_config_has_only_known_keys():
    configs = sorted(CONFIG_DIR.rglob("*.toml"))
    assert configs, "no region configs found"
    for path in configs:
        load_region(path)


def test_missing_section_fails(tmp_path):
    text = VALID.replace("[seasons]", "[not_seasons]")
    path = tmp_path / "region.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[seasons\]"):
        load_region(path)


def test_pseudo_replication_requires_station_regex(tmp_path):
    path = write_config(
        tmp_path,
        **{'obs_unit = "farm"': 'obs_unit = "farm"\npseudo_replicated_rows = true'},
    )
    with pytest.raises(ValueError, match="station_id_regex"):
        load_region(path)


# ---------------------------------------------------------------------------
# The shipped configs are data, and data gets tested: every region file must
# load, and the granularity/hemisphere facts verified in the design review
# are pinned here so a config edit cannot silently un-verify them.
# ---------------------------------------------------------------------------


def shipped(name: str) -> RegionSpec:
    return load_region(CONFIG_DIR / f"{name}.toml")


def test_all_shipped_configs_load():
    paths = sorted(CONFIG_DIR.glob("*.toml"))
    assert len(paths) == 20
    specs = [load_region(p) for p in paths]
    # Codes stay unique even for variant regions over the same country: ES-WS
    # is Spain through WindStats, SE-BZ is Sweden through per-zone observations.
    assert len({s.code for s in specs}) == 20


def test_shipped_granularity_classification():
    assert shipped("dk").obs_unit == "turbine"
    de = shipped("de")
    assert de.obs_unit == "turbine"
    assert de.location_resolution == "postcode"
    uk = shipped("uk")
    assert uk.obs_unit == "farm"
    assert uk.pseudo_replicated_rows is True
    assert uk.station_id_regex
    assert shipped("au_nem").obs_unit == "farm"
    us = shipped("us")
    assert us.obs_unit == "plant"  # EIA-923 reports net generation per plant
    assert us.obs_level == "turbine"  # mechanical branch; the unit is the plant
    br = shipped("br")
    assert br.obs_unit == "complex"  # ONS reports wind per conjunto (complex)
    assert br.obs_level == "turbine"
    nz = shipped("nz")
    assert nz.obs_unit == "farm"  # EMI reports metered injection per plant
    assert nz.obs_level == "turbine"
    cl = shipped("cl")
    assert cl.obs_unit == "plant"  # CEN reports per central (plant)
    assert cl.obs_level == "turbine"
    ar = shipped("ar")
    assert ar.obs_unit == "plant"  # CAMMESA reports per central (plant)
    assert ar.obs_level == "turbine"
    for name in ("nl", "fr", "be", "no", "se", "es", "it", "pt", "ie"):
        spec = shipped(name)
        assert spec.obs_level == "country"
        assert spec.obs_unit == "country"


def test_hemisphere_pin_au_winter_is_jja():
    """AU winter is JJA and UK winter is DJF: the whole point of explicit seasons."""
    au = shipped("au_nem")
    uk = shipped("uk")
    assert sorted(au.seasons["winter"]) == [6, 7, 8]
    assert sorted(uk.seasons["winter"]) == [1, 2, 12]
    assert au.seasons["winter"] != uk.seasons["winter"]
    # Same season NAMES on both sides: transfer matches by name (design §7.3).
    assert set(au.seasons) == set(uk.seasons)


def test_load_region_by_code_resolves_the_stem_and_checks_the_code(tmp_path):
    from pyvwf.harness.regions import load_region_by_code, region_stem

    assert region_stem("AU-NEM") == "au_nem"
    assert load_region_by_code("AU-NEM").code == "AU-NEM"
    # A hyphenated code resolved to a missing file before the lookup was shared.
    assert load_region_by_code("es-ws").code == "ES-WS"
    assert load_region_by_code("nz") == shipped("nz")
    with pytest.raises(FileNotFoundError, match="is 'XX' a shipped region"):
        load_region_by_code("XX")
    # A config whose declared code is not the one asked for is refused.
    (tmp_path / "zz.toml").write_text((CONFIG_DIR / "nz.toml").read_text())
    with pytest.raises(ValueError, match="declares code 'NZ', not 'zz'"):
        load_region_by_code("zz", config_dir=tmp_path)


def test_every_maintained_config_is_named_for_its_code():
    """load_region_by_code finds a config by its file name, so the name is a contract."""
    from pyvwf.harness.regions import load_region_by_code, region_stem

    for path in sorted(CONFIG_DIR.glob("*.toml")):
        spec = load_region(path)
        assert path.stem == region_stem(spec.code), (path.name, spec.code)
        assert load_region_by_code(spec.code) == spec
