"""
Visualization utilities for BKT experiments.

Provides functions to create research-quality plots and figures for:
- Parameter sensitivity curves
- Model comparison charts
- Learning trajectories
- Heatmaps for parameter interactions
- Data efficiency curves
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


def plot_parameter_sensitivity(
    df: pd.DataFrame,
    parameter_name: str,
    metrics: List[str] = ['auc', 'accuracy', 'rmse'],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot sensitivity curve for a single parameter.
    
    Args:
        df: DataFrame from parameter sweep
        parameter_name: Name of parameter
        metrics: Metrics to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Main line
        ax.plot(df['value'], df[metric], 'o-', linewidth=2, markersize=6, 
                label=metric.upper())
        
        # Error bars if available
        if f'{metric}_std' in df.columns:
            ax.fill_between(
                df['value'],
                df[metric] - df[f'{metric}_std'],
                df[metric] + df[f'{metric}_std'],
                alpha=0.2
            )
        
        # Highlight best value
        best_idx = df[metric].idxmax() if metric != 'rmse' else df[metric].idxmin()
        best_value = df.iloc[best_idx]['value']
        best_metric = df.iloc[best_idx][metric]
        ax.axvline(best_value, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.plot(best_value, best_metric, 'r*', markersize=15, label=f'Best: {best_value:.2f}')
        
        ax.set_xlabel(parameter_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} vs {parameter_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_parameter_sensitivities(
    sweep_results: Dict[str, pd.DataFrame],
    save_dir: Optional[str] = None
):
    """
    Create sensitivity plots for all parameters in a single figure.
    """
    params = ['p_init', 'p_learn', 'p_guess', 'p_slip']
    param_labels = ['P(L₀): Initial Knowledge', 'P(T): Learn Rate', 
                    'P(G): Guess Rate', 'P(S): Slip Rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, param, label in zip(axes, params, param_labels):
        if param not in sweep_results:
            continue
        
        df = sweep_results[param]
        
        # Plot AUC
        ax.plot(df['value'], df['auc'], 'o-', linewidth=2.5, markersize=7, 
                color='#2E86AB', label='AUC')
        
        if 'auc_std' in df.columns:
            ax.fill_between(
                df['value'],
                df['auc'] - df['auc_std'],
                df['auc'] + df['auc_std'],
                alpha=0.2,
                color='#2E86AB'
            )
        
        # Mark best
        best_idx = df['auc'].idxmax()
        best_val = df.iloc[best_idx]['value']
        best_auc = df.iloc[best_idx]['auc']
        ax.axvline(best_val, color='#A23B72', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(best_val, best_auc, '*', color='#F18F01', markersize=18)
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax.set_title(f'Sensitivity: {label}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for best value
        ax.annotate(f'Best: {best_val:.2f}\nAUC: {best_auc:.3f}',
                   xy=(best_val, best_auc), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.suptitle('Parameter Sensitivity Analysis - All BKT Parameters', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "all_parameters_sensitivity.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    plt.show()


def plot_parameter_interaction_heatmap(
    df: pd.DataFrame,
    param1_name: str,
    param2_name: str,
    metric: str = 'auc',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create heatmap showing interaction between two parameters.
    """
    # Pivot data for heatmap
    pivot = df.pivot(index=param2_name, columns=param1_name, values=metric)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': metric.upper()},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel(param1_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel(param2_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_title(f'Parameter Interaction: {param1_name} × {param2_name}\n{metric.upper()}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    data: Union[pd.DataFrame, Dict[str, Dict[str, float]]],
    metrics: List[str] = ['test_auc', 'test_accuracy', 'test_rmse'],
    save_path: Optional[str] = None
):
    """
    Create bar chart comparing multiple models.
    Supports both DataFrame (from comparison script) and Dict (from individual runs).
    """
    # Convert Dict to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data).T
        df['model'] = df.index
    else:
        df = data.copy()
        
    # Standardize column names if needed (remove 'test_' prefix if not in df)
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Handle different column naming conventions
        col = metric
        if col not in df.columns and col.replace('test_', '') in df.columns:
            col = col.replace('test_', '')
            
        if col not in df.columns:
            continue
            
        # Sort
        if 'rmse' in col.lower() or 'loss' in col.lower():
            sorted_df = df.sort_values(col)
        else:
            sorted_df = df.sort_values(col, ascending=False)
            
        # Plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[col], 
                     color='#2E86AB', edgecolor='black', linewidth=1.5)
        
        # Add labels
        for i, v in enumerate(sorted_df[col]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
            
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").upper()}', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    # Don't show by default to avoid blocking script execution if run in batch
    plt.close()


def plot_data_efficiency(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Create learning curves showing data efficiency.
    """
    if 'test_auc' not in df.columns or 'sample_size' not in df.columns:
        print("Required columns missing for data efficiency plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by model and sample size
    grouped = df.groupby(['model', 'sample_size'])['test_auc'].agg(['mean', 'std'])
    
    # Plot AUC
    models = df['model'].unique()
    colors = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        if model not in grouped.index: continue
        
        model_data = grouped.loc[model]
        axes[0].plot(model_data.index, model_data['mean'], 'o-', 
                    linewidth=2.5, markersize=8, label=model, color=colors[i])
        axes[0].fill_between(model_data.index,
                            model_data['mean'] - model_data['std'],
                            model_data['mean'] + model_data['std'],
                            alpha=0.2, color=colors[i])
    
    axes[0].set_xlabel('Number of Students', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    axes[0].set_title('Data Efficiency: AUC vs Sample Size', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Training Time (if available)
    # Just counting trials or finding specific time column
    grouped_time = df.groupby(['model', 'sample_size']).size()
    
    for i, model in enumerate(models):
        if model not in df['model'].values: continue
        model_data = df[df['model'] == model]
        sizes = sorted(model_data['sample_size'].unique())
        counts = [len(model_data[model_data['sample_size'] == s]) for s in sizes]
        
        axes[1].plot(sizes, counts, 'o-', linewidth=2.5, 
                    markersize=8, label=model, color=colors[i])
    
    axes[1].set_xlabel('Number of Students', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Trials', fontsize=12, fontweight='bold')
    axes[1].set_title('Trials Completed', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    plt.close()


def plot_learning_trajectory(
    knowledge_states: List[float],
    responses: List[int],
    skill_name: str = "Skill",
    save_path: Optional[str] = None
):
    """
    Visualize knowledge state evolution and responses over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    timesteps = np.arange(len(knowledge_states))
    
    # Plot knowledge state
    ax1.plot(timesteps, knowledge_states, 'o-', linewidth=2.5, markersize=7,
            color='#2E86AB', label='P(Learned)')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Mastery Threshold')
    ax1.fill_between(timesteps, 0, knowledge_states, alpha=0.2, color='#2E86AB')
    ax1.set_ylabel('Probability of Mastery', fontsize=12, fontweight='bold')
    ax1.set_title(f'Learning Trajectory: {skill_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot responses
    colors = ['#D62828' if r == 0 else '#2A9D8F' for r in responses]
    ax2.scatter(timesteps, responses, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Practice Attempt', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Response', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect', 'Correct'])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    
    plt.close()


def create_summary_report_figure(
    recommendations: Dict[str, Dict],
    save_path: str = "results/parameter_recommendations.png"
):
    """
    Create a summary figure showing recommended parameter ranges.
    """
    params = ['p_init', 'p_learn', 'p_guess', 'p_slip']
    labels = ['P(L₀)\nInitial', 'P(T)\nLearn', 'P(G)\nGuess', 'P(S)\nSlip']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(params))
    
    for i, (param, label) in enumerate(zip(params, labels)):
        if param not in recommendations:
            continue
        
        rec = recommendations[param]
        optimal = rec['optimal_value']
        range_min, range_max = rec['recommended_range']
        
        # Plot range as horizontal bar
        ax.barh(i, range_max - range_min, left=range_min, height=0.5,
               color='lightblue', edgecolor='blue', linewidth=2, alpha=0.6,
               label='Good Range' if i == 0 else '')
        
        # Plot optimal value
        ax.plot(optimal, i, 'r*', markersize=20, label='Optimal' if i == 0 else '')
        
        # Add text annotation
        ax.text(optimal + 0.02, i, f'{optimal:.3f}', va='center', fontsize=11,
               fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Parameter Value', fontsize=13, fontweight='bold')
    ax.set_title('Recommended Parameter Values and Ranges', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Summary figure saved to: {save_path}")
    plt.close()
