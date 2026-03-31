"""
research_student_profiling.py
=============================
Phase 4: Student Profiling & Characteristics Analysis (Research Oriented)
This script processes the 5000-student dataset using the best DKT (GRU) model
to extract predicted mastery probabilities. It then classifies students into
pedagogical archetypes (e.g., High Achievers, Careless/Slipping, Lucky Guessers)
by analyzing the discrepancy between predicted mastery and actual answers.

Outputs: 
- 6_student_archetype_distribution.png
- 7_mastery_vs_accuracy_scatter.png
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
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# Set visual style for academic papers
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

def classify_student(accuracy, avg_mastery, slip_rate, guess_rate):
    """
    Adjusted thresholds based on the model's actual confidence distribution.
    """
    # ปรับลดจาก 0.15 เหลือ 0.05 (ถ้าโมเดลมั่นใจแล้วเด็กพลาดแค่ 5% ของข้อสอบ ก็ถือว่าเริ่มสะเพร่าแล้ว)
    if slip_rate >= 0.05:
        return "Careless (High Slip)"
    
    elif guess_rate >= 0.05:
        return "Lucky Guesser"
    
    # ปรับลดความเป็นเด็กเก่งลงมาที่ 0.55 (เพราะ DKT มักจะให้คะแนนเฉลี่ยแถวๆ 0.5)
    elif avg_mastery >= 0.55 and accuracy >= 0.55:
        return "High Achiever"
    
    # ขยับเกณฑ์เด็กอ่อนขึ้นมาที่ 0.45
    elif avg_mastery <= 0.45 and accuracy <= 0.45:
        return "Struggling"
        
    else:
        return "Developing (Average)"

def main():
    parser = argparse.ArgumentParser(description="Generate student profiles for research analysis.")
    parser.add_argument('--data-file', type=str, default='synthetic_data_5000.csv')
    parser.add_argument('--config', type=str, default='best_hyperparameters_5000.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"\n{'='*70}\n 🔬 PHASE 4: RESEARCH STUDENT PROFILING \n{'='*70}")

    # 1. Load Data
    dataset = DataLoader.load_from_csv(
        filepath=args.data_file, col_student='student_id', col_skill='question_id', 
        col_item='question_id', col_correct='is_correct', col_time_taken='response_time'
    )
    num_skills = dataset.num_skills if hasattr(dataset, 'num_skills') else len(set([i.skill_id for seq in dataset.sequences for i in seq.interactions]))
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. Setup and Train GRU (Our best model)
    gru_params = config.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
    gru = DeepKnowledgeTracingGRU(
        num_skills=num_skills, hidden_size=gru_params.get("hidden_size", 128), 
        dropout=gru_params.get("dropout", 0.2), device=args.device
    )
    
    print("⏳ Initializing and training DKT (GRU) Model Engine...")
    gru.fit(dataset, epochs=2, batch_size=gru_params.get("batch_size", 32), learning_rate=gru_params.get("learning_rate", 0.001), verbose=False)

    # 3. Analyze Every Student
    print("🎯 Extracting pedagogical characteristics for all 5,000 students...")
    student_profiles = []

    # Using a subset if 5000 is too slow, but DKT inference is generally fast
    for seq in dataset.sequences:
        total_interactions = len(seq.interactions)
        if total_interactions < 5: 
            continue # Skip students with too little data
            
        history = []
        corrects = 0
        slip_count = 0
        guess_count = 0
        mastery_sum = 0.0
        
        for intr in seq.interactions:
            # Predict probability BEFORE knowing the answer
            p_correct = gru.predict_next(seq.student_id, history, intr.skill_id)
            actual_correct = intr.correct
            
            mastery_sum += p_correct
            corrects += actual_correct
            
            if p_correct >= 0.55 and actual_correct == 0:
                slip_count += 1
            elif p_correct <= 0.45 and actual_correct == 1:
                guess_count += 1
                
            history.append(intr)
            
        accuracy = corrects / total_interactions
        avg_mastery = mastery_sum / total_interactions
        slip_rate = slip_count / total_interactions
        guess_rate = guess_count / total_interactions
        
        profile = classify_student(accuracy, avg_mastery, slip_rate, guess_rate)
        
        student_profiles.append({
            'student_id': seq.student_id,
            'total_interactions': total_interactions,
            'accuracy': accuracy,
            'avg_mastery': avg_mastery,
            'slip_rate': slip_rate,
            'guess_rate': guess_rate,
            'profile': profile
        })

    df = pd.DataFrame(student_profiles)
    
    # 4. Save raw analysis to CSV for paper tables
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "student_profiling_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Raw analysis saved to: {csv_path}")

    # ==========================================
    # 5. Visualization 1: Archetype Distribution
    # ==========================================
    plt.figure(figsize=(9, 7))
    profile_counts = df['profile'].value_counts()
    
    # Define academic-friendly colors
    color_map = {
        "Developing (Average)": "#AAB7B8",
        "High Achiever": "#2ECC71",
        "Careless (High Slip)": "#F1C40F",
        "Struggling": "#E74C3C",
        "Lucky Guesser": "#9B59B6"
    }
    colors = [color_map.get(p, "#333333") for p in profile_counts.index]
    
    plt.pie(profile_counts.values, labels=profile_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.title('Distribution of Student Archetypes Identified by DKT', pad=20, fontweight='bold')
    
    pie_path = output_dir / "6_student_archetype_distribution.png"
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # 6. Visualization 2: Mastery vs Accuracy Scatter
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    ax = sns.scatterplot(
        data=df, x='avg_mastery', y='accuracy', hue='profile', 
        palette=color_map, s=80, alpha=0.7, edgecolor='black'
    )
    
    # Add diagonal line (where Mastery == Accuracy)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=0, label='Mastery = Accuracy (Perfect Calib.)')
    
    plt.title('Predicted Mastery vs. Actual Performance', pad=20, fontweight='bold')
    plt.xlabel('Average Predicted Mastery (Model Confidence)', labelpad=10)
    plt.ylabel('Actual Accuracy (Student Score)', labelpad=10)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    # Place legend cleanly
    plt.legend(title='Student Profile', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    scatter_path = output_dir / "7_mastery_vs_accuracy_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Generated Plot: {pie_path}")
    print(f"✅ Generated Plot: {scatter_path}")
    print(f"\n🎉 PHASE 4 COMPLETE! Ready for Paper Discussion.")

if __name__ == "__main__":
    main()