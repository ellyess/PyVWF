"""Declarative region configs for the validation harness.

A region is one TOML file (``configs/regions/*.toml``) declaring where its
observations come from, which years train and test, where its ERA5 subset
lives, and — critically — its season definitions as explicit month lists.
Seasons are explicit so that Southern-Hemisphere regions cannot silently
inherit the Northern-Hemisphere mapping hardcoded in :mod:`vwf.time_utils`
(the legacy entry points keep that mapping; the harness never uses it).

The loader validates shape and physics up front and fails with the config
path and field in the message. It deliberately does NOT check that
``observations.source`` or ``correction.model`` are registered: configs are
declarative and may name adapters that only exist later (the driver resolves
names against the registries at run time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # tomllib entered the stdlib in 3.11; 3.10 uses the tomli backport.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib  # type: ignore[no-redef]

#: Time slices the pipeline understands (columns produced by
#: vwf.time_utils.add_time_resolution_columns, plus per-month).
VALID_TIME_SLICES = ("fixed", "season", "bimonth", "month")

VALID_OBS_LEVELS = ("turbine", "country")
VALID_OBS_UNITS = ("turbine", "farm", "plant", "complex", "country")


@dataclass(frozen=True)
class RegionSpec:
    """Validated, immutable view of one region config."""

    code: str
    name: str
    source: str
    obs_level: str
    obs_unit: str
    train_years: tuple[int, int]
    test_years: tuple[int, ...]
    era5_path: str
    bbox: tuple[float, float, float, float]
    file_tag: str
    correction_model: str
    cluster_list: tuple[int, ...]
    time_slices: tuple[str, ...]
    seasons: dict[str, tuple[int, ...]] = field(default_factory=dict)
    location_resolution: str | None = None
    pseudo_replicated_rows: bool = False
    station_id_regex: str | None = None
    time_convention: str = "utc-monthly-bins"


def season_of_month(spec: RegionSpec) -> dict[int, str]:
    """Return the month → season-name mapping defined by ``spec``."""
    return {month: name for name, months in spec.seasons.items() for month in months}


def _fail(path: Path, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _require(table: dict, section: str, key: str, path: Path):
    if key not in table:
        _fail(path, f"missing required key [{section}] {key}")
    return table[key]


def _int_list(value, where: str, path: Path) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        _fail(path, f"{where} must be a non-empty list of integers, got {value!r}")
    return tuple(value)


def load_region(path: str | Path) -> RegionSpec:
    """Load and validate a region config.

    Args:
        path: Path to a ``.toml`` region file.

    Returns:
        The validated :class:`RegionSpec`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On any schema or physics violation, naming the field.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    for section in ("region", "observations", "era5", "correction", "seasons"):
        if section not in raw:
            _fail(path, f"missing required section [{section}]")

    region = raw["region"]
    obs = raw["observations"]
    era5 = raw["era5"]
    corr = raw["correction"]
    seasons_raw = raw["seasons"]

    code = str(_require(region, "region", "code", path)).strip()
    name = str(_require(region, "region", "name", path)).strip()
    if not code or not name:
        _fail(path, "[region] code and name must be non-empty")

    source = str(_require(obs, "observations", "source", path)).strip()
    if not source:
        _fail(path, "[observations] source must be non-empty")

    obs_level = _require(obs, "observations", "obs_level", path)
    if obs_level not in VALID_OBS_LEVELS:
        _fail(path, f"[observations] obs_level must be one of {VALID_OBS_LEVELS}, got {obs_level!r}")

    obs_unit = _require(obs, "observations", "obs_unit", path)
    if obs_unit not in VALID_OBS_UNITS:
        _fail(path, f"[observations] obs_unit must be one of {VALID_OBS_UNITS}, got {obs_unit!r}")

    train_years = _int_list(
        _require(obs, "observations", "train_years", path), "[observations] train_years", path
    )
    if len(train_years) != 2 or train_years[0] > train_years[1]:
        _fail(
            path,
            "[observations] train_years must be an inclusive [start, end] pair "
            f"with start <= end, got {list(train_years)}",
        )

    test_years = _int_list(
        _require(obs, "observations", "test_years", path), "[observations] test_years", path
    )
    overlap = [y for y in test_years if train_years[0] <= y <= train_years[1]]
    if overlap:
        _fail(
            path,
            f"[observations] test_years {overlap} fall inside the training window "
            f"{list(train_years)}; held-out validation requires disjoint years",
        )

    pseudo = bool(obs.get("pseudo_replicated_rows", False))
    station_regex = obs.get("station_id_regex")
    if station_regex is not None:
        station_regex = str(station_regex)
    if pseudo and not station_regex:
        _fail(
            path,
            "[observations] pseudo_replicated_rows = true requires station_id_regex "
            "(needed to collapse rows to independent stations)",
        )

    location_resolution = obs.get("location_resolution")
    if location_resolution is not None:
        location_resolution = str(location_resolution)

    time_convention = str(obs.get("time_convention", "utc-monthly-bins"))

    era5_path = str(_require(era5, "era5", "path", path)).strip()
    file_tag = str(_require(era5, "era5", "file_tag", path)).strip()
    if not era5_path or not file_tag:
        _fail(path, "[era5] path and file_tag must be non-empty")

    bbox_raw = _require(era5, "era5", "bbox", path)
    if (
        not isinstance(bbox_raw, list)
        or len(bbox_raw) != 4
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox_raw)
    ):
        _fail(path, f"[era5] bbox must be [lon_min, lon_max, lat_min, lat_max], got {bbox_raw!r}")
    bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
    lon_min, lon_max, lat_min, lat_max = bbox
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        _fail(
            path,
            f"[era5] bbox longitudes must satisfy -180 <= lon_min < lon_max <= 180, got {bbox}",
        )
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        _fail(path, f"[era5] bbox latitudes must satisfy -90 <= lat_min < lat_max <= 90, got {bbox}")

    correction_model = str(_require(corr, "correction", "model", path)).strip()
    if not correction_model:
        _fail(path, "[correction] model must be non-empty")

    cluster_list = _int_list(
        _require(corr, "correction", "cluster_list", path), "[correction] cluster_list", path
    )
    if any(c < 1 for c in cluster_list):
        _fail(path, f"[correction] cluster_list entries must be >= 1, got {list(cluster_list)}")

    slices_raw = _require(corr, "correction", "time_slices", path)
    if not isinstance(slices_raw, list) or not slices_raw:
        _fail(path, "[correction] time_slices must be a non-empty list")
    bad = [s for s in slices_raw if s not in VALID_TIME_SLICES]
    if bad:
        _fail(path, f"[correction] unknown time_slices {bad}; valid: {list(VALID_TIME_SLICES)}")
    time_slices = tuple(str(s) for s in slices_raw)

    seasons: dict[str, tuple[int, ...]] = {}
    for season_name, months in seasons_raw.items():
        if not isinstance(months, list) or not months or not all(
            isinstance(m, int) and not isinstance(m, bool) for m in months
        ):
            _fail(path, f"[seasons] {season_name} must be a non-empty list of month integers")
        seasons[str(season_name)] = tuple(months)

    all_months = sorted(m for months in seasons.values() for m in months)
    if all_months != list(range(1, 13)):
        _fail(
            path,
            "[seasons] month lists must partition 1..12 (each month in exactly one "
            f"season); got {all_months}",
        )

    return RegionSpec(
        code=code,
        name=name,
        source=source,
        obs_level=str(obs_level),
        obs_unit=str(obs_unit),
        train_years=(train_years[0], train_years[1]),
        test_years=test_years,
        era5_path=era5_path,
        bbox=bbox,
        file_tag=file_tag,
        correction_model=correction_model,
        cluster_list=cluster_list,
        time_slices=time_slices,
        seasons=seasons,
        location_resolution=location_resolution,
        pseudo_replicated_rows=pseudo,
        station_id_regex=station_regex,
        time_convention=time_convention,
    )
