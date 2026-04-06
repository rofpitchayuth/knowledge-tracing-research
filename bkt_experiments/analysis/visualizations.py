"""
analysis/visualizations.py
Generates publication-ready plots. 
Each model gets a distinct color.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def assign_model_group(model_name):
    name_lower = model_name.lower()
    if 'bkt' in name_lower: return 'BKT'
    elif 'deep' in name_lower or 'lstm' in name_lower or 'gru' in name_lower: return 'DKT'
    else: return 'Baseline'

def crawl_and_aggregate_results(base_result_dir):
    all_data = []
    base_path = Path(base_result_dir)
    for csv_path in base_path.rglob("comparison_result*.csv"):
        folder_name = csv_path.parent.name
        size_match = re.search(r'(\d+)', folder_name)
        if not size_match: continue
        
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                model_name = row.get('model', 'Unknown')
                all_data.append({
                    'Dataset_Size': int(size_match.group(1)),
                    'Model_Group': assign_model_group(model_name),
                    'Model_Name': model_name,
                    'Test_AUC': row.get('test_auc', row.get('auc', 0.0)),
                    'Training_Time_sec': row.get('training_time_seconds', row.get('training_time', 0.0))
                })
        except Exception: continue

    if not all_data: return None
    master_df = pd.DataFrame(all_data)
    return master_df.groupby(['Dataset_Size', 'Model_Group', 'Model_Name']).mean().reset_index()

def main():
    target_results_folder = PROJECT_ROOT / "results" / "complete_01-4-26_test1" 
    avg_df = crawl_and_aggregate_results(target_results_folder)
    
    if avg_df is None or avg_df.empty:
        print("❌ ไม่พบข้อมูล กรุณาตรวจสอบชื่อโฟลเดอร์")
        return

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ใช้ Color Palette 8 สีสำหรับ 8 โมเดล
    unique_models = avg_df['Model_Name'].unique()
    palette = dict(zip(unique_models, sns.color_palette("husl", len(unique_models))))

    # Plot 1: AUC
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=avg_df, x='Dataset_Size', y='Test_AUC', hue='Model_Name', 
                 style='Model_Name', markers=True, dashes=False, linewidth=2.5, markersize=9, palette=palette)
    plt.title('Data Efficiency: Test AUC by Dataset Size', pad=15, fontweight='bold')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Average Test AUC')
    plt.xticks(sorted(avg_df['Dataset_Size'].unique()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model Name")
    plt.tight_layout()
    plt.savefig(output_dir / "1_auc_efficiency.png", dpi=300)
    plt.close()

    # Plot 2: Time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=avg_df, x='Dataset_Size', y='Training_Time_sec', hue='Model_Name', 
                 style='Model_Name', markers=True, dashes=False, linewidth=2.5, markersize=9, palette=palette)
    plt.title('Computational Cost: Training Time by Dataset Size', pad=15, fontweight='bold')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Average Training Time (Seconds)')
    plt.xticks(sorted(avg_df['Dataset_Size'].unique()))
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model Name")
    plt.tight_layout()
    plt.savefig(output_dir / "2_time_scaling.png", dpi=300)
    plt.close()

    # Plot 3: Trade-off
    max_size = avg_df['Dataset_Size'].max()
    df_max = avg_df[avg_df['Dataset_Size'] == max_size].copy()
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df_max, x='Training_Time_sec', y='Test_AUC', hue='Model_Name', 
                    s=300, palette=palette, edgecolor='black', alpha=0.9)
    plt.xscale('log')
    plt.title(f'Performance vs. Cost Trade-off (Dataset Size: {max_size})', pad=15, fontweight='bold')
    plt.xlabel('Average Training Time (Seconds) - Log Scale')
    plt.ylabel('Average Test AUC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model Name")
    plt.tight_layout()
    plt.savefig(output_dir / "3_tradeoff_scatter.png", dpi=300)
    plt.close()
    
    print("✅ Generate Graph 1-3 (Separate Colors) Success!")

if __name__ == "__main__":
    main()