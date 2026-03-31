import argparse
import pandas as pd
from pathlib import Path
import re
import sys

def extract_size_from_name(name: str) -> int:
    match = re.search(r'\d+', name)
    return int(match.group()) if match else 0

def main():
    parser = argparse.ArgumentParser(description="Aggregate trial results into a final summary.")
    parser.add_argument(
        '--results-dir', 
        type=str, 
        required=True,
        help="Path to the complete results directory (e.g., results/complete_20260330_123456)"
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    
    if not base_dir.exists():
        print(f"[ERROR] Directory '{base_dir}' does not exist.")
        sys.exit(1)

    print(f"\nScanning for result files in: {base_dir} ...")
    
    # 1. Find all comparison_results.csv files recursively
    csv_files = list(base_dir.rglob("comparison_results.csv"))
    
    if not csv_files:
        print("[ERROR] No 'comparison_results.csv' files found in the specified directory.")
        sys.exit(1)

    all_data = []

    # 2. Read and label data
    for file_path in csv_files:
        try:
            # Extract trial number and dataset size from path
            # Expected path: .../trial_1/synthetic_data_500/comparison_results.csv
            parent_folder = file_path.parent.name
            trial_folder = file_path.parent.parent.name
            
            data_size = extract_size_from_name(parent_folder)
            trial_num = extract_size_from_name(trial_folder)
            
            df = pd.read_csv(file_path)
            df['dataset_size'] = data_size
            df['trial'] = trial_num
            
            all_data.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {file_path}: {e}")

    # 3. Combine into a single DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    # 4. Group by Model and Dataset Size, then calculate Mean and Std
    summary_df = master_df.groupby(['dataset_size', 'model']).agg(
        auc_mean=('test_auc', 'mean'),
        auc_std=('test_auc', 'std'),
        time_mean=('training_time_seconds', 'mean')
    ).reset_index()

    # Fill NaN std with 0 (in case there's only 1 trial)
    summary_df['auc_std'] = summary_df['auc_std'].fillna(0)

    # Format the AUC output as "Mean ± Std" for the research paper
    summary_df['auc_formatted'] = summary_df.apply(
        lambda row: f"{row['auc_mean']:.4f} ± {row['auc_std']:.4f}", axis=1
    )

    # 5. Pivot table for display (Models as rows, Dataset Sizes as columns)
    pivot_auc = summary_df.pivot(index='model', columns='dataset_size', values='auc_formatted')
    pivot_time = summary_df.pivot(index='model', columns='dataset_size', values='time_mean')

    # Save the raw aggregated data for future charting
    output_csv = base_dir / "aggregated_final_results.csv"
    summary_df.to_csv(output_csv, index=False)

    # 6. Print the beautiful Markdown table
    print(f"\n{'='*80}")
    print("FINAL CROSS-DATASET EVALUATION RESULTS (3 TRIALS AVERAGE)")
    print(f"{'='*80}")
    print("\n[ TEST AUC (Mean ± Std) ]")
    print("-" * 80)
    print(pivot_auc.to_markdown())
    
    print("\n[ TRAINING TIME (Seconds) ]")
    print("-" * 80)
    print(pivot_time.to_markdown(floatfmt=".2f"))
    print(f"\n{'='*80}")
    print(f"Aggregated data saved to: {output_csv}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()