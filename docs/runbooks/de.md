# Germany (WindStats)

**CONFIDENTIAL / COMMERCIAL DATA.** The German wind data is a licensed
**WindStats** extract, not open data. It (and anything derived from it) must
never be committed or redistributed. `input/` is git-ignored; the script below
only moves files you already hold locally.

Germany uses the shared `european-turbine` adapter, which reads the raw
WindStats layout directly, so there is no transform to reproduce, only a
validate-and-stage step:

```bash
python scripts/process/de.py --src "<path to the WindStats folder>"
python scripts/process/de.py --src "<...>" --check-only   # validate only
```

The three files it stages into `input/observations/turbine/DE/`:

| File | Content |
| --- | --- |
| `DE_md.csv` | Per-turbine metadata: `V1` (id, whose first 5 chars are the German postcode), `Manufacturer`, `kW`, `Rotor..m.`, `Tower..m.` |
| `geolocate.germany.csv` | `postcode -> lon/lat`. Why Germany needs **no external coordinate source**: the postcode embedded in the id geolocates each turbine |
| `DE_data.csv` | Monthly generation: `ID`, `Year`, `Month`, `Output` (kWh), `Downtime` |

On the current extract: **11,433 turbines, 15.7 GW, 88.9% postcode
geolocation coverage, 1998-2019** (the adapter drops turbines that fail to
geolocate). `european-turbine` converts the monthly `Output` to capacity
factor against `kW` at load time; nothing else is needed.

Licence note: unlike the openly-scriptable regions (NZ, CL, AR, DK-register,
UK-REPD), Germany's inputs are commercial. Results computed from them are fine
for research per the WindStats terms, but the data and derived observation
tables stay out of the repository.
