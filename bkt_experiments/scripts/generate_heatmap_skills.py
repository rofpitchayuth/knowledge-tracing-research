"""
generate_skill_heatmap.py
=========================
Phase 2: Deep Analysis (Skill-wise Performance)
This script re-trains the two best models (e.g., Improved BKT vs GRU) on the 5000 dataset,
predicts on the test set, and calculates AUC for EACH specific skill.
It then generates a sorted Heatmap to show where DKT outperforms BKT.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from models.bkt.improved_bkt import ImprovedBKT
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# ─── Smart Parameter Injector ───
def inject_params(model, model_name, config_dict, skills):
    params = config_dict.get(model_name, {}).get("best_params", {})
    if not params: return model

    if isinstance(model, ImprovedBKT):
        logit = lambda p: np.log(p / (1 - p)) if 0 < p < 1 else -2.0
        for skill_id in skills:
            model.params[skill_id] = {
                'p_init': params.get('p_init', 0.5), 'p_learn': params.get('p_learn', 0.1),
                'w_s': 0.0, 'b_s': logit(params.get('p_slip', 0.1)),
                'w_g': 0.0, 'b_g': logit(params.get('p_guess', 0.2))
            }
            model.mean_log_time[skill_id] = 0.0
            model.std_log_time[skill_id] = 1.0
    return model

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    sequences = dataset.sequences
    n = len(sequences)
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    test_idx  = indices[n_train + n_val :] # We only need Train and Test for this analysis
    
    from data.schemas import Dataset as DS
    train_ds = DS(sequences=[sequences[i] for i in train_idx], skills=dataset.skills, items=dataset.items)
    test_ds = DS(sequences=[sequences[i] for i in test_idx], skills=dataset.skills, items=dataset.items)
    return train_ds, test_ds

def get_skill_predictions(model, test_dataset):
    """รัน Inference บน Test Set และเก็บผลลัพธ์แยกตาม Skill"""
    results = []
    for seq in test_dataset.sequences:
        history = []
        for interaction in seq.interactions:
            # Predict before adding to history (to avoid data leakage)
            p_correct = model.predict_next(seq.student_id, history, interaction.skill_id)
            
            results.append({
                'skill_id': interaction.skill_id,
                'true_label': interaction.correct,
                'predicted_prob': p_correct
            })
            history.append(interaction)
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='synthetic_data_5000.csv')
    parser.add_argument('--config', type=str, default='best_hyperparameters_5000.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"\n{'='*70}\n 🔬 PHASE 2: SKILL-WISE PERFORMANCE ANALYSIS \n{'='*70}")
    
    # 1. Load Data & Config
    dataset = DataLoader.load_from_csv(
        filepath=args.data_file, col_student='student_id', col_skill='question_id', 
        col_item='question_id', col_correct='is_correct', col_time_taken='response_time'
    )
    skills = dataset.get_skill_ids() if hasattr(dataset, 'get_skill_ids') else list(dataset.skills)
    num_skills = dataset.num_skills if hasattr(dataset, 'num_skills') else len(skills)
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    train_ds, test_ds = split_dataset(dataset)
    print(f"Data loaded. Train: {train_ds.num_students} students | Test: {test_ds.num_students} students")

    # 2. Setup Models
    bkt = inject_params(ImprovedBKT(), "ImprovedBKT", config, skills)
    
    gru_params = config.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
    gru = DeepKnowledgeTracingGRU(
        num_skills=num_skills, 
        hidden_size=gru_params.get("hidden_size", 128), 
        dropout=gru_params.get("dropout", 0.2), 
        device=args.device
    )

    # 3. Train Models
    print("\n⏳ Training Improved BKT (Fast)...")
    bkt.fit(train_ds, max_iterations=50, verbose=False)
    
    print(f"⏳ Training DKT-GRU on {args.device.upper()} (This will take a few minutes)...")
    gru.fit(train_ds, epochs=15, batch_size=gru_params.get("batch_size", 32), learning_rate=gru_params.get("learning_rate", 0.001), verbose=False)

    # 4. Evaluate per skill
    print("\n🎯 Generating Predictions on Test Set...")
    bkt_preds = get_skill_predictions(bkt, test_ds)
    gru_preds = get_skill_predictions(gru, test_ds)

    skill_auc = []
    for skill in skills:
        bkt_skill_data = bkt_preds[bkt_preds['skill_id'] == skill]
        gru_skill_data = gru_preds[gru_preds['skill_id'] == skill]
        
        # Check if skill has both positive and negative samples (required for AUC)
        if len(bkt_skill_data['true_label'].unique()) > 1:
            b_auc = roc_auc_score(bkt_skill_data['true_label'], bkt_skill_data['predicted_prob'])
            g_auc = roc_auc_score(gru_skill_data['true_label'], gru_skill_data['predicted_prob'])
            skill_auc.append({'Skill': str(skill), 'Improved BKT': b_auc, 'DKT (GRU)': g_auc})

    auc_df = pd.DataFrame(skill_auc)
    
    # Calculate difference to sort the heatmap
    auc_df['Difference (GRU - BKT)'] = auc_df['DKT (GRU)'] - auc_df['Improved BKT']
    
    # Sort by where GRU shines the most
    auc_df = auc_df.sort_values(by='Difference (GRU - BKT)', ascending=False)
    
    # Keep Top 15 and Bottom 15 skills to avoid making the plot too tall and unreadable
    if len(auc_df) > 30:
        plot_df = pd.concat([auc_df.head(15), auc_df.tail(15)])
    else:
        plot_df = auc_df

    # 5. Plot Heatmap
    plt.figure(figsize=(8, 12))
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    # Prepare data for heatmap
    heatmap_data = plot_df.set_index('Skill')[['DKT (GRU)', 'Improved BKT']]
    
    # Draw heatmap
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
        cbar_kws={'label': 'Test AUC'}, linewidths=.5
    )
    
    plt.title("Skill-wise Performance Comparison\n(Sorted by DKT's Advantage)", pad=20, fontweight='bold')
    plt.ylabel("Calculus Skill ID", labelpad=10)
    plt.xlabel("Model", labelpad=10)
    
    plt.tight_layout()
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "4_skill_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ SUCCESS! Heatmap saved to: {save_path}")
    print(f"Insight: Look at the top rows of the heatmap to see which skills DKT handles significantly better!")

if __name__ == "__main__":
    main()