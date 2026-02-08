"""
Create Comparison Visualizations

Generate plots comparing different models.

Usage:
    python create_comparison_plots.py --input results/my_comparison
"""

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing models."""
    metrics = ['test_auc', 'test_accuracy', 'test_rmse']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        # Sort by metric
        if metric == 'test_rmse':
            sorted_df = df.sort_values(metric)
        else:
            sorted_df = df.sort_values(metric, ascending=False)
        
        # Plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric], 
                     color='#2E86AB', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").upper()}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: model_comparison.png")
    plt.close()


def plot_data_efficiency(df: pd.DataFrame, output_dir: Path):
    """Create learning curves showing data efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by model and sample size
    grouped = df.groupby(['model', 'sample_size'])['test_auc'].agg(['mean', 'std'])
    
    # Plot AUC
    for model in df['model'].unique():
        model_data = grouped.loc[model]
        axes[0].plot(model_data.index, model_data['mean'], 'o-', 
                    linewidth=2.5, markersize=8, label=model)
        axes[0].fill_between(model_data.index,
                            model_data['mean'] - model_data['std'],
                            model_data['mean'] + model_data['std'],
                            alpha=0.2)
    
    axes[0].set_xlabel('Number of Students', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    axes[0].set_title('Data Efficiency: AUC vs Sample Size', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Training Time
    grouped_time = df.groupby(['model', 'sample_size']).size()
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        sizes = sorted(model_data['sample_size'].unique())
        counts = [len(model_data[model_data['sample_size'] == s]) for s in sizes]
        
        axes[1].plot(sizes, counts, 'o-', linewidth=2.5, 
                    markersize=8, label=model)
    
    axes[1].set_xlabel('Number of Students', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Trials', fontsize=12, fontweight='bold')
    axes[1].set_title('Data Efficiency Trials', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: data_efficiency.png")
    plt.close()


def main():
    """Create all comparison plots."""
    parser = argparse.ArgumentParser(
        description='Create comparison visualizations'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing comparison results'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    print("\n" + "="*70)
    print(" CREATING COMPARISON VISUALIZATIONS")
    print("="*70)
    print(f"\nInput directory: {input_dir}\n")
    
    # Load and plot comparison results
    results_file = input_dir / 'comparison_results.csv'
    if results_file.exists():
        print("Creating model comparison plot...")
        df = pd.read_csv(results_file)
        plot_model_comparison(df, input_dir)
    else:
        print(f"✗ Missing: {results_file}")
    
    # Load and plot data efficiency
    efficiency_file = input_dir / 'data_efficiency.csv'
    if efficiency_file.exists():
        print("Creating data efficiency plots...")
        df = pd.read_csv(efficiency_file)
        plot_data_efficiency(df, input_dir)
    else:
        print(f"✗ Missing: {efficiency_file}")
    
    print("\n" + "="*70)
    print(" VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nFigures saved to: {input_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
