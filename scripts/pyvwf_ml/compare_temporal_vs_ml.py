#!/usr/bin/env python3
"""Compare temporal split PyVWF results vs ML model results.

Compares:
1. Temporal PyVWF (2015-2019 train, 2023 test) - NEW
2. ML models with unified corrections (random CV) - OLD

Usage:
    python compare_temporal_vs_ml.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Country configurations
COUNTRIES = {
    'NL': {'clusters': 5, 'name': 'Netherlands'},
    'FR': {'clusters': 10, 'name': 'France'},
    'BE': {'clusters': 3, 'name': 'Belgium'},
    'NO': {'clusters': 5, 'name': 'Norway'},
    'ES': {'clusters': 4, 'name': 'Spain'},
    'SE': {'clusters': 4, 'name': 'Sweden'},
    'IT': {'clusters': 3, 'name': 'Italy'},
    'PT': {'clusters': 3, 'name': 'Portugal'},
    'IE': {'clusters': 3, 'name': 'Ireland'},
}

def load_temporal_results():
    """Load temporal split PyVWF results."""
    results = []

    for country, config in COUNTRIES.items():
        run_dir = Path(f"output/temporal_2015_2019_to_2023/output/run/{country}-all-obs_country-corrected-calc_z0")
        cf_dir = run_dir / "results" / "capacity-factor"

        # Load observation data
        obs_train_path = (f"input/country_level_data/observations/{country.lower()}/" +
                          (f"{country.lower()}_train_2015_2019.csv" if country not in ['NO', 'SE']
                           else f"{country.lower()}_train_2015_2019_aggregated.csv"))
        obs_test_path = (f"input/country_level_data/observations/{country.lower()}/" +
                         (f"{country.lower()}_test_2023.csv" if country not in ['NO', 'SE']
                          else f"{country.lower()}_test_2023_aggregated.csv"))

        obs_train = pd.read_csv(obs_train_path)
        obs_train.index = pd.to_datetime(obs_train.iloc[:, 0])

        obs_test = pd.read_csv(obs_test_path)
        obs_test.index = pd.to_datetime(obs_test.iloc[:, 0])

        # Load uncorrected CF
        unc_file = cf_dir / f"{country}_2023_unc_cf.csv"
        if not unc_file.exists():
            print(f"  ⚠ Skipping {country}: uncorrected file not found")
            continue

        unc_cf = pd.read_csv(unc_file, index_col=0, parse_dates=True)

        # Load corrected CF
        cor_file = cf_dir / f"{country}_2023_fixed_{config['clusters']}_cor_cf.csv"
        if not cor_file.exists():
            print(f"  ⚠ Skipping {country}: corrected file not found")
            continue

        cor_cf = pd.read_csv(cor_file, index_col=0, parse_dates=True)

        # Compute metrics
        obs_mean = obs_test['capacity_factor'].mean()
        unc_mean = unc_cf.values.mean()
        cor_mean = cor_cf.values.mean()

        # Errors
        unc_error = abs(unc_mean - obs_mean)
        cor_error = abs(cor_mean - obs_mean)

        # Improvement
        improvement_mean = ((unc_error - cor_error) / unc_error * 100) if unc_error > 0 else 0

        results.append({
            'country': country,
            'name': config['name'],
            'clusters': config['clusters'],
            'obs_train_mean': obs_train['capacity_factor'].mean(),
            'obs_test_mean': obs_mean,
            'unc_mean': unc_mean,
            'cor_mean': cor_mean,
            'unc_error_mean': unc_error,
            'cor_error_mean': cor_error,
            'improvement_mean_pct': improvement_mean,
        })

    return pd.DataFrame(results)

def load_ml_results():
    """Load ML model results (random CV with unified corrections)."""
    # Check for ML results from v2
    ml_file = Path("ml/thesis_outputs_v2/model_comparison_random_cv_summary.csv")

    if not ml_file.exists():
        print("⚠ ML results file not found - checking alternative locations...")
        alt_files = [
            "ml/model_comparison_random_cv_v2/model_comparison_summary.csv",
            "ml/unified_ml/random_cv_results.csv",
        ]
        for alt_file in alt_files:
            if Path(alt_file).exists():
                ml_file = Path(alt_file)
                break

    if not ml_file.exists():
        print("✗ No ML results found for comparison")
        return None

    df = pd.read_csv(ml_file)
    print(f"✓ Loaded ML results from: {ml_file}")
    return df

def main():
    """Main comparison."""
    print("="*80)
    print("TEMPORAL PYVWF vs ML MODEL COMPARISON")
    print("="*80)
    print()

    # Load results
    print("Loading temporal PyVWF results (2015-2019 → 2023)...")
    temporal_df = load_temporal_results()
    print(f"✓ Loaded {len(temporal_df)} countries\n")

    print("Loading ML model results (random CV)...")
    ml_df = load_ml_results()
    print()

    # Display temporal results
    print("="*80)
    print("TEMPORAL PYVWF RESULTS (2015-2019 training → 2023 testing)")
    print("="*80)
    print()

    for _, row in temporal_df.iterrows():
        print(f"{row['name']} ({row['country']}) | {row['clusters']} clusters:")
        print(f"  Training CF (2015-2019): {row['obs_train_mean']:.2%}")
        print(f"  Test CF (2023):          {row['obs_test_mean']:.2%}")
        print(f"  Uncorrected CF:          {row['unc_mean']:.2%} (error: {row['unc_error_mean']:.4f})")
        print(f"  Corrected CF:            {row['cor_mean']:.2%} (error: {row['cor_error_mean']:.4f})")
        print(f"  Mean error improvement:  {row['improvement_mean_pct']:.1f}%")
        print()

    print("="*80)
    print("TEMPORAL PYVWF SUMMARY")
    print("="*80)
    print(f"Countries: {len(temporal_df)}")
    print(f"Mean improvement (mean error): {temporal_df['improvement_mean_pct'].mean():.1f}%")
    print(f"Best country:  {temporal_df.loc[temporal_df['improvement_mean_pct'].idxmax(), 'name']} "
          f"({temporal_df['improvement_mean_pct'].max():.1f}%)")
    print(f"Worst country: {temporal_df.loc[temporal_df['improvement_mean_pct'].idxmin(), 'name']} "
          f"({temporal_df['improvement_mean_pct'].min():.1f}%)")
    print()

    # ML comparison
    if ml_df is not None:
        print("="*80)
        print("ML MODEL RESULTS (Random CV, Unified Corrections)")
        print("="*80)
        print()

        # Show best ML model
        if 'model' in ml_df.columns and 'test_r2' in ml_df.columns:
            best_ml = ml_df.loc[ml_df['test_r2'].idxmax()]
            print(f"Best ML model: {best_ml['model']}")
            print(f"  Test R²:  {best_ml['test_r2']:.3f}")
            print(f"  Test MAE: {best_ml.get('test_mae', 'N/A')}")
            print()

        # Display summary
        print("ML model comparison available in:")
        print(f"  {ml_file}")
        print()

    # Key comparison
    print("="*80)
    print("KEY COMPARISON: TEMPORAL PYVWF vs ML MODELS")
    print("="*80)
    print()
    print("Approach | Method | Performance")
    print("-" * 80)
    print(f"Temporal PyVWF  | Physics-based correction | {temporal_df['improvement_mean_pct'].mean():.1f}% avg improvement")
    if ml_df is not None and 'test_r2' in ml_df.columns:
        best_r2 = ml_df['test_r2'].max()
        print(f"ML Models       | Random Forest (best)     | R² = {best_r2:.3f}")
    print()

    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("✓ Temporal PyVWF: Tests TEMPORAL GENERALIZATION")
    print("  - Trains on 2015-2019, tests on 2023")
    print("  - Evaluates if corrections learned from past work in the future")
    print("  - Accounts for inter-annual wind variability")
    print()

    if ml_df is not None:
        print("✓ ML Models (Random CV): Test INTERPOLATION within dataset")
        print("  - Random split of all correction points")
        print("  - Evaluates if ML can predict corrections at new locations")
        print("  - NOT testing temporal generalization")
        print()

    print("Recommendation:")
    print("  - Use TEMPORAL PYVWF for production (tests real-world scenario)")
    print("  - ML models useful for understanding correction patterns")
    print("="*80)

    # Save comparison
    output_file = Path("output/temporal_vs_ml_comparison.csv")
    temporal_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved temporal results: {output_file}")

    # Save summary
    summary_file = Path("output/temporal_vs_ml_comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL PYVWF vs ML MODEL COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(f"Temporal PyVWF mean improvement: {temporal_df['improvement_mean_pct'].mean():.1f}%\n\n")

        if ml_df is not None and 'test_r2' in ml_df.columns:
            f.write(f"Best ML model R²: {ml_df['test_r2'].max():.3f}\n")

        f.write("\nKey difference:\n")
        f.write("- Temporal PyVWF: TEMPORAL generalization (past → future)\n")
        f.write("- ML models: SPATIAL interpolation (location → location)\n")

    print(f"✓ Saved summary: {summary_file}")

if __name__ == '__main__':
    main()
