import pandas as pd
import numpy as np
import os
import random

# File paths
INPUT_FILE = 'data.xlsx'
OUTPUT_FILE = 'synthetic_data.csv'

def generate_synthetic_data():
    print(f"Reading original data from {INPUT_FILE}...")
    
    try:
        df_q = pd.read_excel(INPUT_FILE, sheet_name='Question')
        df_r = pd.read_excel(INPUT_FILE, sheet_name='Result')
    except Exception as e:
        print(f"Error reading {INPUT_FILE}: {e}")
        return

    # Clean strings
    df_q['correct_answer'] = df_q['correct_answer'].astype(str).str.strip().str.lower()
    df_r['answer'] = df_r['answer'].astype(str).str.strip().str.lower()

    # Merge to calculate correct answers
    df_merged = df_r.merge(df_q[['question_id', 'correct_answer', 'main topic', 'skill_tags']], on='question_id', how='left')
    df_merged['is_correct'] = (df_merged['answer'] == df_merged['correct_answer']).astype(int)

    # Calculate Student Ability (theta) mean and std
    student_accuracy = df_merged.groupby('student_id')['is_correct'].mean()
    mean_ability = student_accuracy.mean()
    std_ability = student_accuracy.std()
    if pd.isna(std_ability) or std_ability == 0:
        std_ability = 0.2  # Add some variance if seed is too homogenous

    # Calculate Question Difficulty (b)
    question_accuracy = df_merged.groupby('question_id')['is_correct'].mean()
    # Invert accuracy to get difficulty
    question_difficulty = 1.0 - question_accuracy 

    print("\n--- Seed Data Analysis ---")
    print(f"Number of Students: {len(student_accuracy)}")
    print(f"Overall Mean Accuracy: {mean_ability:.2%}")
    print(f"Question Difficulties:\n{question_difficulty}")

    # Parameters for generation
    NUM_SYNTHETIC_STUDENTS = 500
    all_questions = df_q['question_id'].tolist()
    
    # Generate Synthetic Students
    print(f"\nGenerating {NUM_SYNTHETIC_STUDENTS} synthetic students...")
    synthetic_records = []
    
    start_id = df_r['student_id'].max() + 1
    if pd.isna(start_id):
        start_id = 1
        
    for i in range(NUM_SYNTHETIC_STUDENTS):
        student_id = start_id + i
        # Sample student true ability (0 to 1)
        # Using a beta distribution or clipped normal based on sample mean
        ability = np.random.normal(mean_ability, std_ability)
        ability = np.clip(ability, 0.05, 0.95)
        
        # Sequence path for this student
        for q_id in all_questions:
            q_diff = question_difficulty.get(q_id, 0.5)
            
            # Simple IRT probability (Rasch model approximation)
            # Prob(correct) increases as ability increases, decreases as difficulty increases
            prob_correct = ability * (1 - q_diff) + (1 - q_diff) * 0.1 
            # Add some noise
            prob_correct = np.clip(prob_correct + np.random.normal(0, 0.1), 0.01, 0.99)
            
            is_correct_simulated = 1 if np.random.random() < prob_correct else 0
            
            # Pick an answer letter based on correctness
            row_q = df_q[df_q['question_id'] == q_id].iloc[0]
            true_ans = row_q['correct_answer']
            
            if is_correct_simulated:
                simulated_ans = true_ans
            else:
                # Pick a random plausible wrong answer (a,b,c,d,e)
                choices = ['a', 'b', 'c', 'd', 'e']
                if true_ans in choices:
                    choices.remove(true_ans)
                simulated_ans = random.choice(choices)
                
            # Simulate response time (log normal roughly)
            sim_time = int(np.random.lognormal(mean=np.log(60), sigma=0.5))
            sim_time = max(5, min(sim_time, 300)) # clip between 5 sec and 300 sec
            
            synthetic_records.append({
                'student_id': student_id,
                'name': f'Synth_Student_{student_id}',
                'question_id': q_id,
                'answer': simulated_ans,
                'response_time': sim_time,
                'is_correct': is_correct_simulated # Added for debugging/direct use
            })

    df_synthetic = pd.DataFrame(synthetic_records)
    
    print("\n--- Synthetic Data Summary ---")
    print(f"Total rows generated: {len(df_synthetic)}")
    print(f"Synthetic Mean Accuracy: {df_synthetic['is_correct'].mean():.2%}")
    
    # Save to file
    df_synthetic.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved synthetic data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_synthetic_data()
