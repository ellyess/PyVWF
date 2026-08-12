# Electricity bidding-zone polygons

Real market-zone geometry for the zonal country-level path. These replace the
approximate bounding boxes hardcoded in
`vwf/datasets/generate_country_level_training_data.py`, which are wrong in ways
that matter: measured against those boxes, 0% of the points in three of the four
Swedish clusters fell inside the zone they were labelled with, and Norway's
NO3/NO5 boundary runs south of Sunnfjord rather than along the Vestland county
line, so a county-shaped box puts Guleslettene, Lutelandet, Hennoy and Mehuken
in the wrong zone.

Files are named `<CC>_<n>.geojson` for numbered zones and `<CC>_<NAME>.geojson`
for Italy's named ones. Cluster index is `n - 1`, matching how
`generate_country_level_training_data` stamps zones onto grid points and how
`EntsoeZonalFileSource` tags observations.

Used by `scripts/region_tools/assign_country_zones.py` and by
`weight_country_grid_points.py --zone-aware`.

## Sources and licences

| Zones | Source | Licence |
|---|---|---|
| `SE_1`..`SE_4`, `DK_1`, `DK_2`, `IT_*` | [EnergieID/entsoe-py](https://github.com/EnergieID/entsoe-py), `entsoe/geo/geojson/` | MIT |
| `NO_1`..`NO_5` | [NVE](https://www.nve.no/karttjenester/) `Nettomraader/MapServer/0` (`Budomraade`) | NLOD 1.0 |

Both permit redistribution with attribution.

`entsoe-py`'s upstreams, per its `entsoe/geo/README.MD`: Natural Earth 10m
(public domain) for single-zone countries, `temakart.nve.no` for Norway,
`natomraden.se` for Sweden, and GME's zone definitions for Italy.

Norway is taken from NVE directly rather than through `entsoe-py` even though
the two are byte-identical to three decimal places, so the citation is
first-party and there is a live update path.

**Deliberately not used:** `electricitymaps-contrib`'s `geo/world.geojson` and
PyPSA-Eur's built bidding zones. The electricityMaps file is AGPL-3.0, which
would be a redistribution problem for this repository, and PyPSA-Eur's output is
derived from it. It is also frozen at Italy's pre-2021 configuration.

Refresh:

```bash
for z in SE_1 SE_2 SE_3 SE_4 DK_1 DK_2 IT_NORD IT_CNOR IT_CSUD IT_SUD IT_SICI IT_SARD IT_CALA; do
  curl -sfL "https://raw.githubusercontent.com/EnergieID/entsoe-py/master/entsoe/geo/geojson/${z}.geojson" -o "${z}.geojson"
done
curl -sfL "https://kart.nve.no/enterprise/rest/services/Nettomraader/MapServer/0/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson" -o nve_all.geojson
```

then split `nve_all.geojson` on its `budomr` attribute (`NO 1` .. `NO 5`, note
the space) into `NO_1.geojson` .. `NO_5.geojson`.

## Caveats

- **Sweden and Denmark are land-only.** Every offshore point falls outside every
  polygon, so callers fall back to the nearest polygon and report how often.
  Norway's polygons do extend offshore.
- **Sweden is the weakest geometry.** `entsoe-py`'s Swedish shapes are a manual
  redraw of a proprietary map (Internaut AB / natomraden.se), and Swedish
  elomrade membership is defined per grid area rather than by a geographic line,
  so no polygon is exactly right. Treat SE as approximate and disclose it.
- **Denmark's DK2 is missing Bornholm** (`DK_2` stops at 12.675 E; Bornholm is
  near 14.9 E). Bornholm is in DK2 despite being cabled to SE4. Not patched here
  because no Danish zonal run exists yet; patch from Natural Earth 10m admin-1
  (public domain) if one is added.
- **Italy changed on 2021-01-01**: CALA was carved out of SUD and Umbria moved
  CNOR to CSUD. The files here are the post-2021 configuration. `entsoe-py` also
  ships `IT_*_2020` variants for earlier dates, not vendored here.
- **Zone stability:** SE1-SE4 unchanged since 2011, NO1-NO5 since 2010, DK1/DK2
  unchanged. The ENTSO-E Bidding Zone Review could change Nordic zones later this
  decade, so pin the file version alongside any published result.
- **Zone code spellings differ**: NVE writes `NO 1` (with a space), `entsoe-py`
  writes `NO_1`, ENTSO-E EIC uses `NO1` / `10Y1001A1001A48H`. Normalise on ingest.
