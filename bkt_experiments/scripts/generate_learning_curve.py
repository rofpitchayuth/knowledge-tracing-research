"""
generate_learning_curve.py
==========================
Phase 2: Qualitative Analysis (Individual Learning Curve)
This script finds a representative student-skill sequence and plots 
the step-by-step predictions of Improved BKT vs DKT (GRU) 
against the student's actual responses.

Output: 5_learning_curve.png
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

def find_candidate_sequence(dataset):
    """
    Finds a student-skill pair that has a good number of interactions (e.g., 6-15)
    and a mix of correct and incorrect answers to make an interesting plot.
    """
    for seq in dataset.sequences:
        # Group interactions by skill
        skill_groups = {}
        for intr in seq.interactions:
            if intr.skill_id not in skill_groups:
                skill_groups[intr.skill_id] = []
            skill_groups[intr.skill_id].append(intr)
            
        for skill_id, interactions in skill_groups.items():
            if 6 <= len(interactions) <= 20:
                corrects = sum(1 for i in interactions if i.correct == 1)
                # Ensure it's not all 1s or all 0s
                if 0 < corrects < len(interactions):
                    return seq.student_id, skill_id, interactions
                    
    # Fallback if no perfect match found
    print("Could not find ideal mixed sequence, picking the longest available.")
    longest_seq = []
    best_student = None
    best_skill = None
    for seq in dataset.sequences:
        skill_groups = {}
        for intr in seq.interactions:
            if intr.skill_id not in skill_groups: skill_groups[intr.skill_id] = []
            skill_groups[intr.skill_id].append(intr)
        for skill_id, interactions in skill_groups.items():
            if len(interactions) > len(longest_seq):
                longest_seq = interactions
                best_student = seq.student_id
                best_skill = skill_id
    return best_student, best_skill, longest_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='synthetic_data_5000.csv')
    parser.add_argument('--config', type=str, default='best_hyperparameters_5000.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"\n{'='*70}\n 🔬 PHASE 2: INDIVIDUAL LEARNING CURVE \n{'='*70}")
    
    # 1. Load Data & Config
    dataset = DataLoader.load_from_csv(
        filepath=args.data_file, col_student='student_id', col_skill='question_id', 
        col_item='question_id', col_correct='is_correct', col_time_taken='response_time'
    )
    skills = dataset.get_skill_ids() if hasattr(dataset, 'get_skill_ids') else list(dataset.skills)
    num_skills = dataset.num_skills if hasattr(dataset, 'num_skills') else len(skills)
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. Find a good candidate to visualize
    target_student, target_skill, target_interactions = find_candidate_sequence(dataset)
    print(f"Selected Candidate -> Student: {target_student} | Skill: {target_skill} | Interactions: {len(target_interactions)}")

    # 3. Setup and Train Models (We must train them on the dataset first)
    bkt = inject_params(ImprovedBKT(), "ImprovedBKT", config, skills)
    
    gru_params = config.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
    gru = DeepKnowledgeTracingGRU(
        num_skills=num_skills, 
        hidden_size=gru_params.get("hidden_size", 128), 
        dropout=gru_params.get("dropout", 0.2), 
        device=args.device
    )

    print("\n⏳ Training Models to generate predictions (This may take a few minutes)...")
    bkt.fit(dataset, max_iterations=20, verbose=False)
    gru.fit(dataset, epochs=10, batch_size=gru_params.get("batch_size", 32), learning_rate=gru_params.get("learning_rate", 0.001), verbose=False)

    # 4. Generate Step-by-Step Predictions for the Target
    print("\n🎯 Generating Step-by-Step Predictions...")
    bkt_probs = []
    gru_probs = []
    actual_answers = []
    
    # We need the full student history up to this point, but for simplicity
    # we simulate passing the interactions one by one
    history = []
    for intr in target_interactions:
        # Predict probability of getting THIS interaction correct, based on history
        p_bkt = bkt.predict_next(target_student, history, target_skill)
        p_gru = gru.predict_next(target_student, history, target_skill)
        
        bkt_probs.append(p_bkt)
        gru_probs.append(p_gru)
        actual_answers.append(intr.correct)
        
        # Add to history for the next step
        history.append(intr)

    # 5. Plotting
    steps = np.arange(1, len(target_interactions) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Model Predictions (Lines)
    plt.plot(steps, bkt_probs, marker='s', linestyle='--', linewidth=2, markersize=8, color='steelblue', label='Improved BKT Prediction')
    plt.plot(steps, gru_probs, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='crimson', label='DKT (GRU) Prediction')
    
    # Plot Actual Responses (Scatter points)
    # We plot correct (1.0) and incorrect (0.0)
    actual_colors = ['red' if a == 0 else 'green' for a in actual_answers]
    plt.scatter(steps, actual_answers, c=actual_colors, s=200, edgecolor='black', zorder=5, label='Actual Answer (Green=Correct, Red=Incorrect)')
    
    # Formatting
    plt.title(f'Tracking Student Learning Trajectory\n(Student ID: {target_student} | Skill ID: {target_skill})', pad=20, fontweight='bold')
    plt.xlabel('Interaction Attempt Number', labelpad=10)
    plt.ylabel('Probability of Correct Answer P(Correct)', labelpad=10)
    
    plt.ylim(-0.1, 1.1)
    plt.xticks(steps)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Custom Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Ensure scatter legend is clear
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "5_learning_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ SUCCESS! Learning Curve saved to: {save_path}")

if __name__ == "__main__":
    main()