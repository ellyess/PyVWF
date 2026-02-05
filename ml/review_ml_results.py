#!/usr/bin/env python3
"""
Review ML Training Results and Diagnostic Plots

This script provides a comprehensive review of the ML model training results,
including diagnostic plots, performance metrics, feature importance, and
prediction maps.

Usage:
    python review_ml_results.py [--model-dir output/ml_europe]
"""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_model_data(model_path):
    """Load trained model and metadata."""
    print(f"Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("\nModel Information:")
    print(f"  Training countries: {', '.join(model_data['training_countries'])}")
    print(f"  Model type: {model_data['model_type']}")
    print(f"  Targets trained: {', '.join(model_data['models'].keys())}")
    
    return model_data


def display_performance_metrics(model_data):
    """Display performance metrics in a table."""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    
    performance = model_data['performance']
    
    # Create DataFrame for nice display
    metrics_data = []
    for target, metrics in performance.items():
        metrics_data.append({
            'Target': target.capitalize(),
            'R² Score': f"{metrics['r2']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
        })
    
    df = pd.DataFrame(metrics_data)
    print("\n" + df.to_string(index=False))
    
    # Interpretation
    print("\n" + "-"*70)
    print("Interpretation:")
    for target, metrics in performance.items():
        r2 = metrics['r2']
        print(f"\n{target.capitalize()}:")
        if r2 > 0.7:
            quality = "Excellent"
            note = "Model explains >70% of variance"
        elif r2 > 0.5:
            quality = "Good"
            note = "Model explains >50% of variance"
        elif r2 > 0.3:
            quality = "Moderate"
            note = "Model has predictive power but limited"
        else:
            quality = "Poor"
            note = "Model struggles to predict this target"
        
        print(f"  R² = {r2:.4f} → {quality} ({note})")
        print(f"  Average error (MAE): {metrics['mae']:.4f}")


def plot_feature_importance(model_data, output_dir):
    """Create feature importance plots."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    n_targets = len(model_data['models'])
    fig, axes = plt.subplots(1, n_targets, figsize=(7*n_targets, 6))
    
    if n_targets == 1:
        axes = [axes]
    
    for ax, (target, model_info) in zip(axes, model_data['models'].items()):
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            
            # Top 15 features
            n_show = min(15, len(feature_cols))
            top_indices = indices[:n_show]
            
            # Plot
            ax.barh(range(n_show), importance[top_indices], color='steelblue', alpha=0.7)
            ax.set_yticks(range(n_show))
            ax.set_yticklabels([feature_cols[i] for i in top_indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {target.capitalize()}')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Print to console
            print(f"\nTop 10 Features for {target.capitalize()}:")
            for i, idx in enumerate(top_indices[:10], 1):
                print(f"  {i:2d}. {feature_cols[idx]:25s} {importance[idx]:.4f}")
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'feature_importance.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n✓ Saved feature importance plot: {output_path}")
    plt.close()


def plot_prediction_maps(predictions_nc, output_dir):
    """Create maps of ML predictions."""
    print("\n" + "="*70)
    print("PREDICTION MAPS")
    print("="*70)
    
    if not Path(predictions_nc).exists():
        print(f"  Prediction file not found: {predictions_nc}")
        return
    
    print(f"  Loading predictions: {predictions_nc}")
    ds = xr.open_dataset(predictions_nc)
    
    # Get variables
    variables = list(ds.data_vars)
    n_vars = len(variables)
    
    print(f"  Variables: {', '.join(variables)}")
    
    # Create figure
    fig, axes = plt.subplots(1, n_vars, figsize=(8*n_vars, 6), subplot_kw={'projection': None})
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        data = ds[var]
        
        # Plot
        im = ax.contourf(
            ds.lon, ds.lat, data,
            levels=20,
            cmap='RdYlBu_r',
            extend='both'
        )
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'ML Predicted {var.capitalize()}\n(DK, UK, DE trained)')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(f'{var.capitalize()} Correction Factor')
        
        # Add gridlines
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Statistics
        print(f"\n{var.capitalize()} statistics:")
        print(f"  Mean: {float(data.mean()):.4f}")
        print(f"  Std:  {float(data.std()):.4f}")
        print(f"  Min:  {float(data.min()):.4f}")
        print(f"  Max:  {float(data.max()):.4f}")
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'prediction_maps.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n✓ Saved prediction maps: {output_path}")
    plt.close()


def create_summary_report(model_data, predictions_nc, output_dir):
    """Create comprehensive summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    output_path = Path(output_dir) / 'ML_TRAINING_SUMMARY.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ML TRAINING SUMMARY REPORT\n")
        f.write("PyVWF - European Correction Factor Prediction\n")
        f.write("="*70 + "\n\n")
        
        # Model info
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Training countries: {', '.join(model_data['training_countries'])}\n")
        f.write(f"Model type: {model_data['model_type']}\n")
        f.write(f"Targets: {', '.join(model_data['models'].keys())}\n")
        f.write("\n")
        
        # Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        for target, metrics in model_data['performance'].items():
            f.write(f"\n{target.capitalize()}:\n")
            f.write(f"  R² Score: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE:     {metrics['rmse']:.4f}\n")
            f.write(f"  MAE:      {metrics['mae']:.4f}\n")
            
            # Interpretation
            r2 = metrics['r2']
            if r2 > 0.7:
                f.write(f"  Quality:  Excellent (explains {r2*100:.1f}% of variance)\n")
            elif r2 > 0.5:
                f.write(f"  Quality:  Good (explains {r2*100:.1f}% of variance)\n")
            elif r2 > 0.3:
                f.write(f"  Quality:  Moderate (explains {r2*100:.1f}% of variance)\n")
            else:
                f.write(f"  Quality:  Poor (explains only {r2*100:.1f}% of variance)\n")
        
        f.write("\n")
        
        # Feature importance
        f.write("FEATURE IMPORTANCE (Top 10)\n")
        f.write("-"*70 + "\n")
        for target, model_info in model_data['models'].items():
            model = model_info['model']
            feature_cols = model_info['feature_cols']
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1]
                
                f.write(f"\n{target.capitalize()}:\n")
                for i, idx in enumerate(indices[:10], 1):
                    f.write(f"  {i:2d}. {feature_cols[idx]:25s} {importance[idx]:.4f}\n")
        
        f.write("\n")
        
        # Predictions
        if Path(predictions_nc).exists():
            ds = xr.open_dataset(predictions_nc)
            f.write("PREDICTION STATISTICS\n")
            f.write("-"*70 + "\n")
            for var in ds.data_vars:
                data = ds[var]
                f.write(f"\n{var.capitalize()}:\n")
                f.write(f"  Mean:  {float(data.mean()):.4f}\n")
                f.write(f"  Std:   {float(data.std()):.4f}\n")
                f.write(f"  Range: [{float(data.min()):.4f}, {float(data.max()):.4f}]\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*70 + "\n\n")
        
        # Automatic insights
        scalar_r2 = model_data['performance'].get('scalar', {}).get('r2', 0)
        
        if scalar_r2 > 0.5:
            f.write("✓ Model successfully learns terrain → correction relationships\n")
            f.write("  → Can extrapolate to regions without training data\n")
        else:
            f.write("⚠ Model has limited predictive power\n")
            f.write("  → Terrain features may not fully explain corrections\n")
            f.write("  → Consider adding more features (land cover, climate zones, etc.)\n")
        
        f.write("\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*70 + "\n")
        
        if scalar_r2 > 0.7:
            f.write("• Model performance is excellent\n")
            f.write("• Safe to use predictions for European-wide applications\n")
        elif scalar_r2 > 0.5:
            f.write("• Model performance is good\n")
            f.write("• Can use predictions with moderate confidence\n")
            f.write("• Consider validating against independent data where available\n")
        else:
            f.write("• Model performance is limited\n")
            f.write("• Use predictions cautiously\n")
            f.write("• Consider:\n")
            f.write("  - Adding more training countries\n")
            f.write("  - Including additional features (land cover, climate)\n")
            f.write("  - Using ensemble models or different ML algorithms\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Saved summary report: {output_path}")


def display_all_plots(output_dir):
    """Display all generated plots."""
    print("\n" + "="*70)
    print("DIAGNOSTIC PLOTS")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # Find all PNG files
    plot_files = sorted(output_dir.glob('*.png'))
    
    if not plot_files:
        print("  No plot files found")
        return
    
    print(f"\nFound {len(plot_files)} plots:")
    for i, plot_file in enumerate(plot_files, 1):
        print(f"  {i}. {plot_file.name}")
    
    print("\nPlot files saved in:", output_dir)
    print("\nTo view plots:")
    print(f"  open {output_dir}")
    print("  or use your favorite image viewer")


def main():
    parser = argparse.ArgumentParser(
        description='Review ML training results and diagnostic plots',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='ml_europe',
        help='Directory containing model and results'
    )
    parser.add_argument(
        '--open-plots',
        action='store_true',
        help='Open plots directory in system viewer (macOS/Linux)'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        print("  Run: python train_europe_ml_corrections.py")
        sys.exit(1)
    
    print("="*70)
    print("ML TRAINING RESULTS REVIEW")
    print("="*70)
    print(f"Model directory: {model_dir}")
    print("="*70)
    
    # Load model
    model_path = model_dir / 'europe_correction_model.pkl'
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        sys.exit(1)
    
    model_data = load_model_data(model_path)
    
    # Display performance metrics
    display_performance_metrics(model_data)
    
    # Plot feature importance
    plot_feature_importance(model_data, model_dir)
    
    # Plot prediction maps
    predictions_nc = model_dir / 'europe_corrections_ml.nc'
    plot_prediction_maps(predictions_nc, model_dir)
    
    # Create summary report
    create_summary_report(model_data, predictions_nc, model_dir)
    
    # Display plots info
    display_all_plots(model_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("REVIEW COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved in: {model_dir}")
    print("\nGenerated files:")
    print(f"  ✓ {model_dir / 'europe_correction_model.pkl'}")
    print(f"  ✓ {model_dir / 'europe_corrections_ml.nc'}")
    print(f"  ✓ {model_dir / 'scalar_predictions.png'}")
    print(f"  ✓ {model_dir / 'feature_importance.png'}")
    print(f"  ✓ {model_dir / 'prediction_maps.png'}")
    print(f"  ✓ {model_dir / 'ML_TRAINING_SUMMARY.txt'}")
    
    if args.open_plots:
        import subprocess
        try:
            subprocess.run(['open', str(model_dir)])
            print(f"\n✓ Opened {model_dir} in system viewer")
        except Exception as e:
            print(f"\n✗ Could not open directory: {e}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
