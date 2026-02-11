"""
Create Visualizations from Experiment Results.

This script creates publication-quality visualizations for:
1. Model Comparison (BKT vs Logistic vs DKT)
2. Data Efficiency Analysis
3. Parameter Sensitivity Analysis
4. Parameter Interaction Heatmaps

Usage:
    python create_visualizations.py --input results/my_experiment
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

from analysis.visualizations import (
    plot_model_comparison,
    plot_data_efficiency,
    plot_all_parameter_sensitivities,
    plot_parameter_interaction_heatmap,
    create_summary_report_figure
)


def main():
    """Create visualizations from experiment results."""
    parser = argparse.ArgumentParser(
        description='Create visualizations from BKT experiment results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing experiment results (csv files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for figures (default: INPUT/figures)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if args.output is None:
        output_dir = input_dir / "figures"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" CREATING VISUALIZATIONS")
    print("="*70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 1. Model Comparison Plots
    # ---------------------------------------------------------
    results_file = input_dir / 'comparison_results.csv'
    if results_file.exists():
        print("Creating model comparison plots...")
        df = pd.read_csv(results_file)
        plot_model_comparison(
            df, 
            save_path=str(output_dir / 'model_comparison.png')
        )
        print(f"  ✓ Saved: model_comparison.png")
    
    # 2. Data Efficiency Plots
    # ---------------------------------------------------------
    efficiency_file = input_dir / 'data_efficiency.csv'
    if efficiency_file.exists():
        print("\nCreating data efficiency plots...")
        df = pd.read_csv(efficiency_file)
        plot_data_efficiency(
            df,
            save_path=str(output_dir / 'data_efficiency.png')
        )
        print(f"  ✓ Saved: data_efficiency.png")
        
    # 3. Parameter Sensitivity Plots
    # ---------------------------------------------------------
    print("\nChecking for parameter sweep results...")
    sweeps = {}
    for param in ['p_init', 'p_learn', 'p_guess', 'p_slip']:
        csv_file = input_dir / f"sweep_{param}.csv"
        if csv_file.exists():
            sweeps[param] = pd.read_csv(csv_file)
            print(f"  ✓ Loaded {param} sweep")
    
    if sweeps:
        print("Creating parameter sensitivity plots...")
        plot_all_parameter_sensitivities(sweeps, save_dir=str(output_dir))
        print(f"  ✓ Saved: all_parameters_sensitivity.png")
    
    # 4. Interaction Heatmaps
    # ---------------------------------------------------------
    print("\nChecking for interaction results...")
    
    # P(L0) vs P(T)
    interaction_file = input_dir / "interaction_p_init_vs_p_learn.csv"
    if interaction_file.exists():
        df = pd.read_csv(interaction_file)
        plot_parameter_interaction_heatmap(
            df, 'p_init', 'p_learn',
            save_path=str(output_dir / "interaction_init_vs_learn.png"),
            show=False
        )
        print(f"  ✓ Saved: interaction_init_vs_learn.png")
    
    # P(G) vs P(S)
    interaction_file = input_dir / "interaction_p_guess_vs_p_slip.csv"
    if interaction_file.exists():
        df = pd.read_csv(interaction_file)
        plot_parameter_interaction_heatmap(
            df, 'p_guess', 'p_slip',
            save_path=str(output_dir / "interaction_guess_vs_slip.png"),
            show=False
        )
        print(f"  ✓ Saved: interaction_guess_vs_slip.png")
    
    # 5. Recommendations Summary
    # ---------------------------------------------------------
    recs_file = input_dir / "parameter_recommendations.json"
    if recs_file.exists():
        print("\nCreating recommendations summary...")
        with open(recs_file, 'r') as f:
            recommendations = json.load(f)
        
        create_summary_report_figure(
            recommendations,
            save_path=str(output_dir / "recommendations_summary.png")
        )
        print(f"  ✓ Saved: recommendations_summary.png")
    
    print("\n" + "="*70)
    print(" VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
