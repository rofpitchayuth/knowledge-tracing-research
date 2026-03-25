"""
Create Visualizations from Experiment Results.

This script creates publication-quality visualizations for:
1. Model Comparison (BKT vs Logistic vs DKT)
2. Data Efficiency Analysis
3. Parameter Sensitivity Analysis
4. Parameter Interaction Heatmaps

Usage:
    python create_visualizations.py --input results/complete_TIMESTAMP
"""

import argparse
import pandas as pd
from pathlib import Path
import json

from analysis.visualizations import (
    plot_model_comparison,
    plot_data_efficiency,
    plot_all_parameter_sensitivities,
    plot_parameter_interaction_heatmap,
    create_summary_report_figure,
    plot_cross_dataset_efficiency
)

def process_single_experiment(input_dir: Path, base_output_dir: Path):
    """
    Process a single experiment directory containing csv results.
    """
    if input_dir.name != base_output_dir.parent.name:
        output_dir = base_output_dir / input_dir.name
    else:
        output_dir = base_output_dir
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'-'*60}")
    print(f" PROCESSING: {input_dir.name}")
    print(f"{'-'*60}")
    
    # 1. Model Comparison Plots
    # ---------------------------------------------------------
    results_file = input_dir / 'comparison_results.csv'
    if results_file.exists():
        print("  -> Creating model comparison plots...")
        df = pd.read_csv(results_file)
        plot_model_comparison(
            df, 
            save_path=str(output_dir / 'model_comparison.png')
        )
    else:
        print(f"  -> Skipping: No 'comparison_results.csv' found in {input_dir.name}")
        return # ข้ามโฟลเดอร์นี้ไปถ้าไม่มีผลลัพธ์หลัก

    # 2. Data Efficiency Plots
    # ---------------------------------------------------------
    efficiency_file = input_dir / 'data_efficiency.csv'
    if efficiency_file.exists():
        print("  -> Creating data efficiency plots...")
        df = pd.read_csv(efficiency_file)
        plot_data_efficiency(
            df,
            save_path=str(output_dir / 'data_efficiency.png')
        )
        
    # 3. Parameter Sensitivity Plots
    # ---------------------------------------------------------
    sweeps = {}
    for param in ['p_init', 'p_learn', 'p_guess', 'p_slip']:
        csv_file = input_dir / f"sweep_{param}.csv"
        if csv_file.exists():
            sweeps[param] = pd.read_csv(csv_file)
            
    if sweeps:
        print("  -> Creating parameter sensitivity plots...")
        plot_all_parameter_sensitivities(sweeps, save_dir=str(output_dir))
    
    # 4. Interaction Heatmaps
    # ---------------------------------------------------------
    interaction_file1 = input_dir / "interaction_p_init_vs_p_learn.csv"
    if interaction_file1.exists():
        print("  -> Creating interaction heatmaps (init vs learn)...")
        df = pd.read_csv(interaction_file1)
        plot_parameter_interaction_heatmap(
            df, 'p_init', 'p_learn',
            save_path=str(output_dir / "interaction_init_vs_learn.png"),
            show=False
        )
    
    interaction_file2 = input_dir / "interaction_p_guess_vs_p_slip.csv"
    if interaction_file2.exists():
        print("  -> Creating interaction heatmaps (guess vs slip)...")
        df = pd.read_csv(interaction_file2)
        plot_parameter_interaction_heatmap(
            df, 'p_guess', 'p_slip',
            save_path=str(output_dir / "interaction_guess_vs_slip.png"),
            show=False
        )
    
    # 5. Recommendations Summary
    # ---------------------------------------------------------
    recs_file = input_dir / "parameter_recommendations.json"
    if recs_file.exists():
        print("  -> Creating recommendations summary...")
        with open(recs_file, 'r') as f:
            recommendations = json.load(f)
        
        create_summary_report_figure(
            recommendations,
            save_path=str(output_dir / "recommendations_summary.png")
        )
    
    print(f"  ✓ Saved all figures for {input_dir.name} to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create visualizations from BKT experiment results (Supports multiple sizes)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing experiment results (e.g., results/complete_TIMESTAMP)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for figures (default: INPUT/figures)'
    )
    
    args = parser.parse_args()
    
    base_input_dir = Path(args.input)
    if args.output is None:
        base_output_dir = base_input_dir / "figures"
    else:
        base_output_dir = Path(args.output)
        
    print("\n" + "="*70)
    print(" CREATING VISUALIZATIONS")
    print("="*70)
    print(f"Base Input directory:  {base_input_dir}")
    print(f"Base Output directory: {base_output_dir}\n")
    
    if not base_input_dir.exists():
        print(f"Error: Input directory {base_input_dir} does not exist.")
        return

    # ตรวจสอบว่าเป็นโฟลเดอร์ที่มีไฟล์ผลลัพธ์โดยตรง หรือเป็นโฟลเดอร์แม่ที่มีโฟลเดอร์ย่อย
    size_results = {}  # dict: dataset_size -> DataFrame (used for cross-dataset efficiency)

    if (base_input_dir / 'comparison_results.csv').exists():
        # แบบเก่า: มีไฟล์ csv อยู่ในนี้เลย
        process_single_experiment(base_input_dir, base_output_dir)
    else:
        # แบบใหม่: ค้นหาโฟลเดอร์ย่อยทั้งหมด (เช่น synthetic_data_500, 1000)
        # ข้ามโฟลเดอร์ที่ชื่อว่า 'figures'
        subdirs = [d for d in base_input_dir.iterdir() if d.is_dir() and d.name != "figures"]
        
        if not subdirs:
            print(f"No experiment subdirectories found in {base_input_dir}")
            return
            
        print(f"Found {len(subdirs)} dataset sizes. Processing each...")
        
        for subdir in sorted(subdirs):
            process_single_experiment(subdir, base_output_dir)

            # Collect results for cross-dataset efficiency chart
            results_file = subdir / 'comparison_results.csv'
            if results_file.exists():
                # Extract dataset size from folder name (e.g. synthetic_data_500 -> 500)
                try:
                    size = int(subdir.name.split('_')[-1])
                    size_results[size] = pd.read_csv(results_file)
                except (ValueError, IndexError):
                    pass  # Skip folders whose name doesn't end in a number

        # --- Cross-Dataset Efficiency Chart ---
        if len(size_results) > 1:
            print(f"\n  -> Creating cross-dataset efficiency chart ({len(size_results)} sizes)...")
            base_output_dir.mkdir(parents=True, exist_ok=True)
            plot_cross_dataset_efficiency(
                size_results,
                save_path=str(base_output_dir / 'data_efficiency.png')
            )
        else:
            print("  -> Skipping cross-dataset efficiency chart (need at least 2 dataset sizes).")


    print("\n" + "="*70)
    print(" VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"All figures have been saved under: {base_output_dir}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()