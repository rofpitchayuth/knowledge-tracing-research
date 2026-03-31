import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import numpy as np

# Set visual style for academic papers
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Cannot find {csv_path}. Please run aggregate_results.py first.")
        sys.exit(1)
    return pd.read_csv(path)

def plot_auc_efficiency(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 6))
    
    # We will use markers to clearly show the 3 data points
    ax = sns.lineplot(
        data=df, x='dataset_size', y='auc_mean', hue='model', 
        marker='o', markersize=8, linewidth=2.5, err_style=None
    )
    
    # Format axes
    plt.title('Knowledge Tracing Performance vs. Data Scale', pad=20, fontweight='bold')
    plt.xlabel('Dataset Size (Number of Interactions)', labelpad=10)
    plt.ylabel('Test AUC (Mean)', labelpad=10)
    plt.xscale('log') # Log scale because sizes are 500, 1000, 5000
    
    # Set exact ticks for the X axis
    dataset_sizes = sorted(df['dataset_size'].unique())
    plt.xticks(dataset_sizes, dataset_sizes)
    
    # Customize Legend
    plt.legend(title='KT Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = output_dir / '1_auc_efficiency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_time_scaling(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Training Time vs Dataset Size"""
    plt.figure(figsize=(10, 6))
    
    ax = sns.lineplot(
        data=df, x='dataset_size', y='time_mean', hue='model', 
        marker='s', markersize=8, linewidth=2.5, err_style=None
    )
    
    plt.title('Training Time Scaling (Computational Cost)', pad=20, fontweight='bold')
    plt.xlabel('Dataset Size (Number of Interactions)', labelpad=10)
    plt.ylabel('Training Time (Seconds) - Log Scale', labelpad=10)
    plt.xscale('log')
    plt.yscale('log') # Log scale for Y because DKT is much slower
    
    dataset_sizes = sorted(df['dataset_size'].unique())
    plt.xticks(dataset_sizes, dataset_sizes)
    
    plt.legend(title='KT Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = output_dir / '2_time_scaling.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path):
    """Plot 3: AUC vs Time Trade-off (Only for the largest dataset, with Legend)"""
    largest_size = df['dataset_size'].max()
    df_largest = df[df['dataset_size'] == largest_size].copy()
    
    plt.figure(figsize=(10, 7))
    
    ax = sns.scatterplot(
        data=df_largest, x='time_mean', y='auc_mean', hue='model', 
        style='model', s=400, markers=True, legend="full"
    )
    
    # Set the labels and log scale
    plt.title(f'Performance vs. Cost Trade-off (Dataset Size: {largest_size})', pad=20, fontweight='bold')
    plt.xlabel('Training Time (Seconds) - Log Scale', labelpad=10)
    plt.ylabel('Test AUC', labelpad=10)
    plt.xscale('log')
    
    plt.legend(title='KT Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = output_dir / '3_tradeoff_scatter_fixed.png' # Add '_fixed' to show the difference
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready plots from aggregated results.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to the aggregated_final_results.csv file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    df = load_data(str(input_path))
    
    plot_auc_efficiency(df, output_dir)
    plot_time_scaling(df, output_dir)
    plot_tradeoff_scatter(df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Figures saved in: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()