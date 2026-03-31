"""
generate_learning_curve_v2.py
=============================
Phase 2: Qualitative Analysis (Cross-Skill Learning Trajectory)
Plots the chronological sequence of a student's answers across DIFFERENT skills.
Highlights how DKT uses past interactions from other skills to inform future predictions,
while BKT resets for each new skill.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from models.bkt.improved_bkt import ImprovedBKT
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# Set visual style
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='synthetic_data_5000.csv')
    parser.add_argument('--config', type=str, default='best_hyperparameters_5000.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 1. Load Data
    dataset = DataLoader.load_from_csv(
        filepath=args.data_file, col_student='student_id', col_skill='question_id', 
        col_item='question_id', col_correct='is_correct', col_time_taken='response_time'
    )
    skills = dataset.get_skill_ids() if hasattr(dataset, 'get_skill_ids') else list(dataset.skills)
    num_skills = dataset.num_skills if hasattr(dataset, 'num_skills') else len(skills)
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. Pick a student (Take the first student with at least 20 interactions)
    target_student = None
    target_interactions = []
    for seq in dataset.sequences:
        if len(seq.interactions) >= 20:
            target_student = seq.student_id
            target_interactions = seq.interactions[:20] # เอา 20 ข้อแรกมาวิเคราะห์
            break

    # 3. Setup Models
    bkt = inject_params(ImprovedBKT(), "ImprovedBKT", config, skills)
    gru_params = config.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
    gru = DeepKnowledgeTracingGRU(
        num_skills=num_skills, hidden_size=gru_params.get("hidden_size", 128), 
        dropout=gru_params.get("dropout", 0.2), device=args.device
    )

    print("\n⏳ Training Models (Fast setup)...")
    bkt.fit(dataset, max_iterations=5, verbose=False) # ไม่ต้องเทรนลึกเพราะเรามี params แล้ว
    gru.fit(dataset, epochs=2, batch_size=gru_params.get("batch_size", 32), learning_rate=gru_params.get("learning_rate", 0.001), verbose=False)

    # 4. Predict
    print("\n🎯 Generating Predictions...")
    bkt_probs, gru_probs, actual_answers, skill_labels = [], [], [], []
    history = []
    
    for intr in target_interactions:
        p_bkt = bkt.predict_next(target_student, history, intr.skill_id)
        p_gru = gru.predict_next(target_student, history, intr.skill_id)
        
        bkt_probs.append(p_bkt)
        gru_probs.append(p_gru)
        actual_answers.append(intr.correct)
        skill_labels.append(f"S{intr.skill_id}")
        
        history.append(intr)

    # 5. Plotting
    steps = np.arange(1, len(target_interactions) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.plot(steps, bkt_probs, marker='s', linestyle='--', color='steelblue', label='Improved BKT', linewidth=2, markersize=8)
    plt.plot(steps, gru_probs, marker='o', linestyle='-', color='crimson', label='DKT (GRU)', linewidth=2.5, markersize=8)
    
    colors = ['red' if a == 0 else 'green' for a in actual_answers]
    plt.scatter(steps, actual_answers, c=colors, s=150, edgecolor='black', zorder=5, label='Actual Answer (Green=Correct)')
    
    plt.title(f'Cross-Skill Learning Trajectory (Student: {target_student})', pad=20, fontweight='bold')
    plt.xlabel('Chronological Question Attempt (with Skill ID)', labelpad=10)
    plt.ylabel('Predicted Probability P(Correct)', labelpad=10)
    
    plt.ylim(-0.1, 1.1)
    plt.xticks(steps, skill_labels, rotation=45) # โชว์ว่าแต่ละข้อคือ Skill อะไร
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    save_path = Path("paper_figures/5_cross_skill_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ SUCCESS! Saved to: {save_path}")

if __name__ == "__main__":
    main()