# ENTSO-E API Integration Guide

Complete guide for fetching wind generation and capacity factor data from the ENTSO-E Transparency Platform API.

## Quick Start

### 1. Installation

```bash
pip install entsoe-py pandas pyarrow
```

### 2. Get API Key

1. Register at: https://transparency.entsoe.eu/
2. Login → Account Settings → Web API Security Token
3. Generate and copy your token

### 3. Set Environment Variable

```bash
export ENTSOE_API_KEY='your-api-key-here'
```

Or add to your `.bashrc` / `.zshrc` for persistence:

```bash
echo 'export ENTSOE_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Run the Script

#### Command Line Interface

```bash
# Fetch hourly capacity factors for NL, FR, BE (2015-2020)
python scripts/fetch_entsoe_capacity_factors.py \
    --countries NL FR BE \
    --year-start 2015 \
    --year-end 2020 \
    --resample h \
    --output-dir output/entsoe_data

# Fetch daily data for a single country
python scripts/fetch_entsoe_capacity_factors.py \
    --countries NL \
    --year-start 2019 \
    --year-end 2019 \
    --resample D \
    --format csv

# Fetch offshore wind only
python scripts/fetch_entsoe_capacity_factors.py \
    --countries NL \
    --year-start 2020 \
    --year-end 2020 \
    --psr-type offshore
```

#### Python API

```python
import pandas as pd
from scripts.fetch_entsoe_capacity_factors import ENTSOEWindDataFetcher

# Initialize
fetcher = ENTSOEWindDataFetcher(api_key="your-key-or-None-for-env-var")

# Fetch capacity factors
start = pd.Timestamp("2020-01-01", tz="UTC")
end = pd.Timestamp("2020-12-31", tz="UTC")

df = fetcher.calculate_capacity_factor(
    country="NL",
    start=start,
    end=end,
    psr_type="all"  # 'onshore', 'offshore', or 'all'
)

print(df.head())
#                      generation_mw  capacity_mw  capacity_factor
# 2020-01-01 00:00:00         1234.5       4500.0         0.274333
# 2020-01-01 01:00:00         1456.2       4500.0         0.323600
# ...
```

## Available Countries

| Country | Code | Bidding Zones | Data Coverage | Notes |
|---------|------|---------------|---------------|-------|
| **Netherlands** | `NL` | Single zone | ✅ Full | Onshore + Offshore |
| **France** | `FR` | Single zone | ✅ Full | Onshore + Offshore |
| **Belgium** | `BE` | Single zone | ✅ Full | Onshore + Offshore |
| **Norway** | `NO` | 5 zones | ✅ Full | Automatically aggregates NO_1 to NO_5 |

### Norway Bidding Zones

Norway is divided into 5 bidding zones in ENTSO-E:

| Zone Code | Region | Description |
|-----------|--------|-------------|
| `NO_1` | Oslo | Eastern Norway |
| `NO_2` | Kristiansand | Southern Norway |
| `NO_3` | Trondheim | Mid-Norway |
| `NO_4` | Tromsø | Northern Norway |
| `NO_5` | Bergen | Western Norway |

**Usage:**
- Use `--countries NO` to automatically fetch and aggregate all 5 zones
- Or specify individual zones: `--countries NO_1 NO_3` for specific regions

---

## Fetching Turbine Metadata

In addition to capacity factors, you can extract turbine/unit metadata from ENTSO-E:

### Quick Start

```bash
# Fetch metadata for NL, FR, BE, NO
python scripts/fetch_entsoe_turbine_metadata.py \
    --countries NL FR BE NO \
    --year 2020

# Fetch only Norwegian zones
python scripts/fetch_entsoe_turbine_metadata.py \
    --countries NO \
    --year 2021
```

### What Metadata is Available?

The ENTSO-E API provides:
- ✅ **Unit EIC codes** (unique identifiers)
- ✅ **Unit names** (plant/facility names)
- ✅ **Estimated capacity** (derived from max generation)
- ✅ **Type** (onshore/offshore classification)
- ✅ **Country/Zone** information

**⚠️ Limitations:**
- ❌ No location data (lon/lat coordinates)
- ❌ No hub height information
- ❌ No rotor diameter
- ❌ No manufacturer information

For spatial modeling, you'll need to match ENTSO-E units with other metadata sources that include locations.

### Output Format

```text
ID,capacity,type,name,country,zone
NL001234567890123456,3000.0,onshore,WindPark Alpha,NL,
NO_1_987654321,2500.0,offshore,Hywind Beta,NO,NO_1
```

### Combining with Capacity Factors

```bash
# 1. Fetch metadata
python scripts/fetch_entsoe_turbine_metadata.py \
    --countries NL \
    --year 2020 \
    --output-dir output/nl_data

# 2. Fetch capacity factors
python scripts/fetch_entsoe_capacity_factors.py \
    --countries NL \
    --year-start 2020 \
    --year-end 2020 \
    --output-dir output/nl_data

# Now you have both:
# - output/nl_data/nl_turbine_metadata.csv
# - output/nl_data/nl_capacity_factors.csv
```

## Output Formats

### 1. Hourly Time Series (Default)

```text
timestamp,generation_mw,capacity_mw,capacity_factor,country
2020-01-01 00:00:00,1234.5,4500.0,0.274333,NL
2020-01-01 01:00:00,1456.2,4500.0,0.323600,NL
...
```

### 2. Daily Aggregated (`--resample D`)

```text
timestamp,generation_mw,capacity_mw,capacity_factor,country
2020-01-01,1345.2,4500.0,0.299,NL
2020-01-02,2103.4,4500.0,0.467,NL
...
```

### 3. Monthly PyVWF Format

```python
from examples.entsoe_api_examples import example_monthly_aggregation

# Produces wide format for PyVWF:
#       1       2       3  ...      12
# 2015  0.234   0.267   0.301  ...  0.189
# 2016  0.221   0.289   0.312  ...  0.203
# ...
```

## Examples

Run the comprehensive examples:

```bash
python examples/entsoe_api_examples.py
```

This will demonstrate:
1. Fetching data for a single country
2. Fetching data for multiple countries (NL, FR, BE)
3. Saving data to CSV/Parquet files
4. Comparing onshore vs offshore capacity factors
5. Creating monthly aggregated data (PyVWF format)

## API Parameters

### `--countries` (CLI) / `country` (Python)
Country codes: `NL`, `FR`, `BE`, `NO` (limited)

### `--psr-type` (CLI) / `psr_type` (Python)
- `all` - Onshore + Offshore (default)
- `onshore` - Onshore wind only
- `offshore` - Offshore wind only

### `--resample` (CLI) / `resample` (Python)
- `h` - Hourly (raw data, ~15-min resolution upsampled)
- `D` - Daily average
- `M` - Monthly average
- `None` - No resampling (Python only)

### `--format` (CLI)
- `csv` - CSV files
- `parquet` - Parquet files (more efficient)
- `both` - Both formats (default)

## Understanding the Data

### Generation Data
- **Source**: Actual Generation Output per Generation Unit (A73)
- **Resolution**: 15-minute intervals (aggregated to hourly by default)
- **Units**: MW (Megawatts)

### Capacity Data
- **Source**: Installed Generation Capacity per Unit (A68)
- **Updates**: Yearly or when capacity changes
- **Units**: MW (Megawatts)

### Capacity Factor Calculation

```python
capacity_factor = generation_mw / capacity_mw
```

**Interpretation:**
- `0.25` = 25% capacity factor (typical for onshore wind)
- `0.40` = 40% capacity factor (typical for offshore wind)
- `0.50+` = 50%+ capacity factor (excellent wind conditions)

**Note**: Values can occasionally exceed 1.0 due to:
- Short-term overgeneration
- Capacity data being outdated
- Data quality issues

The script clips values at 1.5 to handle these edge cases.

## Combining with PyVWF

### Use Case 1: Training Bias Corrections

```python
from vwf.data import train_set
from scripts.fetch_entsoe_capacity_factors import ENTSOEWindDataFetcher

# 1. Fetch observations from ENTSO-E
fetcher = ENTSOEWindDataFetcher()
df = fetcher.calculate_capacity_factor(
    country="NL",
    start=pd.Timestamp("2015-01-01", tz="UTC"),
    end=pd.Timestamp("2020-12-31", tz="UTC"),
)

# 2. Save in PyVWF format
# Convert to monthly wide format and save to input/country-data/NL/observations/

# 3. Run PyVWF
gen_cf, turb_info, reanalysis, power_curves = train_set(
    country="NL",
    obs_level="country",  # Use country-level CF from ENTSO-E
    year_test=None,
)
```

### Use Case 2: Validation

```python
# Fetch actual generation for validation period
actual_2021 = fetcher.calculate_capacity_factor(
    country="NL",
    start=pd.Timestamp("2021-01-01", tz="UTC"),
    end=pd.Timestamp("2021-12-31", tz="UTC"),
)

# Compare with PyVWF predictions
# ... your validation code here ...
```

## Troubleshooting

### Error: "No matching data"

**Cause**: Country or time period not available in ENTSO-E.

**Solutions:**
1. Check country code is correct (NL, FR, BE)
2. Try a different date range (some countries have data from ~2015 onwards)
3. For Norway, use alternative sources

### Error: "401 Unauthorized"

**Cause**: Invalid or missing API key.

**Solutions:**
1. Verify your API key is correct
2. Check the environment variable is set: `echo $ENTSOE_API_KEY`
3. Try passing the key directly: `--api-key your-key`

### Error: "Too many requests"

**Cause**: Rate limiting (400 requests per minute).

**Solutions:**
1. Reduce the date range
2. Add delays between requests
3. Use caching (the script doesn't re-fetch existing data)

### Warning: "Capacity factor > 1.0"

**Cause**: Temporary overgeneration or outdated capacity data.

**Solutions:**
- The script automatically clips at 1.5
- For analysis, you can filter: `df = df[df['capacity_factor'] <= 1.0]`
- Or use median/percentile instead of mean

## Data Quality Notes

1. **Missing Data**: Some timestamps may be missing (outages, maintenance)
2. **Capacity Updates**: Installed capacity changes infrequently (yearly)
3. **Aggregation**: Generation is reported per unit, capacity per unit
4. **Timezone**: All timestamps are UTC by default
5. **Resolution**: Original data is 15-minute, upsampled to hourly

## Advanced Usage

### Custom Resampling

```python
# Fetch raw data
df = fetcher.calculate_capacity_factor("NL", start, end)

# Custom resampling (weekly, with specific aggregations)
weekly = df.resample("W").agg({
    "generation_mw": "sum",      # Total weekly generation
    "capacity_mw": "last",        # Last capacity value
    "capacity_factor": ["mean", "std", "max"]  # Multiple stats
})
```

### Batch Processing Multiple Years

```bash
# Fetch each year separately to avoid rate limiting
for year in {2015..2020}; do
    python scripts/fetch_entsoe_capacity_factors.py \
        --countries NL FR BE \
        --year-start $year \
        --year-end $year \
        --output-dir output/entsoe_data/$year
    sleep 5  # Delay between requests
done
```

### Combining with Local Data

```python
# Fetch ENTSO-E data
entsoe_data = fetcher.calculate_capacity_factor("NL", start, end)

# Load local turbine metadata
from vwf.data import load_turbine_metadata
nl_meta = load_turbine_metadata("NL")

# Merge or compare
# ... your analysis code ...
```

## Support

- **ENTSO-E API Docs**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- **entsoe-py Documentation**: https://github.com/EnergieID/entsoe-py
- **PyVWF Issues**: [Your repo issues]

## References

- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
- Zenodo Static Dataset: https://zenodo.org/records/4682697 (2014-2021, 5 countries)
- This implementation: Dynamic fetching via API (all ENTSO-E countries, up-to-date)
