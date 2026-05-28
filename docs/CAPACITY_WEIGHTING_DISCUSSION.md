# Capacity Weighting Discussion and Design Decisions

## Overview

This document summarizes the discussion and decisions around capacity weighting in PyVWF bias correction calculations, covering both turbine-level and country-level workflows.

## The Core Question

**Should bias correction factors (scalars and offsets) be calculated using capacity-weighted or unweighted averaging?**

This affects:
1. **Scalar calculation**: The multiplicative correction factor per cluster
2. **Offset optimization**: The additive wind speed adjustment per cluster
3. **Final aggregation**: Combining clusters to country-level output

## Key Principles Established

### 1. Scalars Should Represent Spatial Meteorological Bias

**Physical reasoning:**
- A scalar of 1.15 means "reanalysis underestimates wind by 15% in this spatial region"
- This is a property of the location/meteorology, NOT turbine size or capacity distribution
- If you moved a 1MW turbine to where a 5MW turbine is, it should get the same scalar

**Why capacity weighting is problematic:**
- Conflates spatial meteorological patterns with fleet composition
- Makes scalars depend on "what size turbines happen to be built there" rather than "what's the reanalysis error there"
- Breaks transferability: can't apply corrections to new locations with different capacity distributions

**Implications for ML/Interpolation:**
- ML models learn: `terrain/location features → bias correction`
- Capacity weighting adds noise: same features could yield different scalars based on turbine mix
- Unweighted scalars enable clean feature-to-bias learning

### 2. Different Purpose: Bias Correction vs Energy Matching

**Scalar (multiplicative):**
- Purpose: Represent systematic spatial bias in reanalysis
- Should be: Unweighted (pure spatial property)
- Used for: Spatial transferability, ML feature learning

**Offset (additive):**
- Purpose: Fine-tune to match aggregate cluster energy output
- Can be: Capacity-weighted (matches total energy)
- Used for: Accurate energy totals at validation

**Final Aggregation:**
- Purpose: Combine corrected clusters to country total
- Must be: Capacity-weighted (economically relevant total)
- Used for: Country-level generation estimates

## Implementation Decisions

### Scalar Calculation

**Location:** `src/vwf/correction.py::calculate_scalar()`

**Decision: UNWEIGHTED for all workflows**

```python
# Simple mean aggregation (no capacity weighting)
df = gen_cf.groupby([time_res, 'cluster', 'year']).agg({
    "obs": "mean",
    "sim": "mean",
})
```

**Rationale:**
- Consistent across turbine-level and country-level
- Scalars represent pure spatial bias
- Better for ML/interpolation methods
- Previous capacity-weighted code commented out for reference

### Offset Optimization

**Location:** `src/vwf/correction.py::find_offset()` and `find_offsets_country_level()`

**Uses:** `src/vwf/wind.py::train_simulate_wind()`

**Decision: CAPACITY-WEIGHTED (kept as-is)**

```python
# In train_simulate_wind
avg_cf = cor_cf.weighted(cor_cf['capacity']).mean()
```

**Rationale:**
- Offset tunes aggregate cluster output to match observations
- Capacity weighting ensures offset optimizes for total energy, not per-turbine average
- Consistent with physical goal: "adjust wind by Y m/s to match cluster's total energy"
- Different conceptual purpose than scalar (energy matching vs spatial bias)

**Note:** This means:
- **Turbine-level**: Capacity weighting reflects real turbine sizes → matches cluster energy
- **Country-level**: Capacity weighting reflects grid point capacities → may favor high-capacity regions

### Final Aggregation

**Location:** `src/vwf/vwf.py::simulate_cf()` calls `src/vwf/wind.py::simulate_country_cf()`

**Decision: CAPACITY-WEIGHTED (always required)**

```python
# In simulate_country_cf
w = sim_cf["capacity"]
country_cf = sim_cf.weighted(w).mean("turbine")
```

**Rationale:**
- Necessary to get correct country-level generation
- Each cluster/turbine contributes proportionally to its capacity
- Economically meaningful total output

## Year-Specific Grid Point Weighting

**Decision: NOT RECOMMENDED**

### The Issue

Year-specific weighting means grid points change based on historical capacity installation patterns:
- More grid points where/when capacity was high
- Fewer grid points where/when capacity was low
- Conflates "where should we sample the weather?" with "where were turbines built?"

### Evidence from Evaluation

**With year-specific weighting (`country_grid_2015_2021_2023`):**
- Netherlands: R² = -0.93 (catastrophic failure)
- Norway: R² = 0.21 (was 0.54 without weighting)
- Portugal: R² = -0.14 (went negative)
- 5 out of 9 countries degraded

**Without year-specific weighting (`all_grid`):**
- Better overall performance
- Netherlands: R² = 0.06 (still trained successfully)
- Norway: R² = 0.54 (much better)

### Recommended Approach

**Use static grid points:**
```python
'use_year_specific_weighting': False
```

**What this means:**
- Grid points represent spatial sampling of meteorological conditions
- Each grid point gets capacity from a single representative year (e.g., 2023 or 2015-2021 average)
- Scalars represent pure spatial bias without capacity growth patterns
- Better for ML/interpolation (spatial features → bias, without year-to-year capacity noise)
- More transferable to future scenarios (PyPSA-Eur with different spatial distributions)

### When Year-Specific Weighting Might Make Sense

Year-specific weighting would be appropriate if:
- Reanalysis quality varied with fleet composition (it doesn't)
- You're analyzing how build-out evolution affected systematic errors (not the main goal)
- Historical hindcasting is more important than future transferability (opposite of PyPSA-Eur use case)
- Training only to validate past years, not predict future scenarios

## Practical Rule for Pipeline Choice

For **PyVWF → PyPSA-Eur** thesis pipeline:

### Training Corrections for Historical Validation
- **Purpose**: Reproduce actual past generation
- **Approach**: Can use capacity weighting if needed
- **Use case**: Validate against ENTSO-E 2015-2023 data

### Training Corrections for Future Scenarios
- **Purpose**: Transfer bias corrections to new locations/configurations
- **Approach**: Unweighted scalars, static grid points
- **Use case**: PyPSA-Eur 2030/2040/2050 with different spatial distributions

**Recommendation:** Focus on approach #2 (future transferability) as primary method, with #1 available for sensitivity analysis.

## Summary of Final Configuration

| Component | Weighting | Rationale |
|-----------|-----------|-----------|
| **Scalar calculation** | Unweighted | Pure spatial bias, ML-friendly |
| **Offset optimization** | Capacity-weighted | Matches aggregate energy output |
| **Final aggregation** | Capacity-weighted | Correct country-level totals |
| **Grid point selection** | Static (not year-specific) | Transferability, better performance |

## Code Locations

**Modified files:**
- correction.py: `calculate_scalar()` - unweighted aggregation
- data.py: `cluster_train_set()` - removed capacity weighting for country-level
- train_all_bias_corrections.py: Added `use_year_specific_weighting` toggle

**Unchanged (capacity weighting retained):**
- wind.py: `train_simulate_wind()` - capacity-weighted CF for offset optimization
- wind.py: `simulate_country_cf()` - capacity-weighted final aggregation

## References

Old capacity-weighted code is preserved as comments in correction.py for:
- Comparison experiments
- Sensitivity analysis
- Thesis discussion of methodological choices

---

**Date:** 2025-02-12
**Status:** Design decisions implemented and tested

Save this as `CAPACITY_WEIGHTING_DISCUSSION.md` in your project root directory.Save this as `CAPACITY_WEIGHTING_DISCUSSION.md` in your project root directory.