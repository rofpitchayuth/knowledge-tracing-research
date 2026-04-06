"""
scripts/research_student_profiling.py
=====================================
Uses mathematical distance from the diagonal line (Mastery = Accuracy) 
to classify students.
"""
import argparse, json, torch, sys
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.data_loader import DataLoader
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def classify_student(accuracy, avg_mastery):
    """ใช้ระยะห่างจากเส้นทแยงมุมเป็นตัวแบ่งกลุ่มอย่างชัดเจน"""
    diff = accuracy - avg_mastery
    
    # 1. เดาเก่ง: คะแนนสูงกว่าความรู้จริงที่ AI ประเมินอย่างน้อย 10%
    if diff >= 0.10: return "Lucky Guesser"
    
    # 2. สะเพร่า: ความรู้จริงสูงกว่าคะแนนสอบอย่างน้อย 10%
    elif diff <= -0.10: return "Careless (High Slip)"
    
    # ถ้าอยู่ใกล้เส้นประ (Diff < 10%) แปลว่าคะแนนสะท้อนความสามารถจริง
    # ให้ดูที่เกณฑ์ความเก่ง
    elif avg_mastery >= 0.55 and accuracy >= 0.55: return "High Achiever"
    elif avg_mastery <= 0.45 and accuracy <= 0.45: return "Struggling"
    else: return "Developing (Average)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/synthetic_data_5000_diverse.csv')
    parser.add_argument('--config', type=str, default='data/best_hyperparameters_5000.json')
    parser.add_argument('--output-dir', type=str, default='results/complete_01-4-26_test1/figures')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = DataLoader.load_from_csv(str(PROJECT_ROOT / args.data_file), 'student_id', 'question_id', 'question_id', 'is_correct', 'response_time')
    skills = list(dataset.skills) if hasattr(dataset, 'skills') else []
    
    with open(PROJECT_ROOT / args.config, 'r') as f:
        config = json.load(f)
    p = config.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
    
    gru = DeepKnowledgeTracingGRU(num_skills=len(skills), hidden_size=p.get("hidden_size", 128), dropout=p.get("dropout", 0.2), device='cpu')
    gru.fit(dataset, epochs=2, batch_size=p.get("batch_size", 32), learning_rate=p.get("learning_rate", 0.001), verbose=False)

    profiles = []
    for seq in dataset.sequences:
        if len(seq.interactions) < 5: continue
        history, corrects, mastery_sum = [], 0, 0.0
        for intr in seq.interactions:
            mastery_sum += gru.predict_next(seq.student_id, history, intr.skill_id)
            corrects += intr.correct
            history.append(intr)
            
        acc = corrects / len(seq.interactions)
        mast = mastery_sum / len(seq.interactions)
        profiles.append({'avg_mastery': mast, 'accuracy': acc, 'profile': classify_student(acc, mast)})

    df = pd.DataFrame(profiles)
    
    color_map = {"Developing (Average)": "#AAB7B8", "High Achiever": "#2ECC71", 
                 "Careless (High Slip)": "#F1C40F", "Struggling": "#E74C3C", "Lucky Guesser": "#9B59B6"}
                 
    # Plot 4: Pie Chart
    plt.figure(figsize=(9, 7))
    counts = df['profile'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[color_map.get(p) for p in counts.index])
    plt.title('Distribution of Student Archetypes', pad=20, fontweight='bold')
    plt.savefig(output_dir / "4_student_archetype_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Scatter
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='avg_mastery', y='accuracy', hue='profile', palette=color_map, s=80, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Mastery = Accuracy')
    plt.title('Mastery vs. Actual Performance', pad=20, fontweight='bold')
    plt.xlabel('Predicted Mastery'); plt.ylabel('Actual Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(output_dir / "5_mastery_vs_accuracy_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generate Profiling Graphs Success!")

if __name__ == "__main__":
    main()