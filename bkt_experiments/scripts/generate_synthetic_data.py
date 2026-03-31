import pandas as pd
import numpy as np
import os
import random

# File paths
INPUT_FILE = 'data.xlsx'

def generate_synthetic_data(sample_sizes):
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

    # Setup parameters for generation
    all_questions = df_q['question_id'].tolist()
    start_id = df_r['student_id'].max() + 1
    if pd.isna(start_id):
        start_id = 1
        
    # =========================================================
    # Loop through each sample size to generate multiple datasets
    # =========================================================
    for size in sample_sizes:
        print(f"\n=========================================")
        print(f"Generating synthetic data for {size} students...")
        print(f"=========================================")
        
        synthetic_records = []
        
        for i in range(size):
            # To ensure unique IDs across different generated sets if needed, 
            # or just start from the same start_id for each separate file.
            student_id = start_id + i
            
            # Sample student true ability (0 to 1)
            ability = np.random.normal(mean_ability, std_ability)
            ability = np.clip(ability, 0.05, 0.95)
            
            # Sequence path for this student
            for q_id in all_questions:
                q_diff = question_difficulty.get(q_id, 0.5)
                
                # Simple IRT probability (Rasch model approximation)
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
                    # Pick a random plausible wrong answer
                    choices = ['a', 'b', 'c', 'd', 'e']
                    if true_ans in choices:
                        choices.remove(true_ans)
                    if not choices: # Safeguard
                        choices = ['a', 'b', 'c']
                    simulated_ans = random.choice(choices)
                    
                # Simulate response time
                sim_time = int(np.random.lognormal(mean=np.log(60), sigma=0.5))
                sim_time = max(5, min(sim_time, 300)) # clip between 5 sec and 300 sec
                
                synthetic_records.append({
                    'student_id': student_id,
                    'name': f'Synth_Student_{student_id}',
                    'question_id': q_id,
                    'answer': simulated_ans,
                    'response_time': sim_time,
                    'is_correct': is_correct_simulated
                })

        # Create DataFrame for this specific size
        df_synthetic = pd.DataFrame(synthetic_records)
        
        # Save to file with dynamic naming (e.g., synthetic_data_500.csv)
        output_file = f'synthetic_data_{size}.csv'
        df_synthetic.to_csv(output_file, index=False)
        
        print(f"Total rows generated: {len(df_synthetic)}")
        print(f"Synthetic Mean Accuracy: {df_synthetic['is_correct'].mean():.2%}")
        print(f"Saved synthetic data to: {output_file}")

if __name__ == "__main__":
    # Define the target sample sizes here
    target_sample_sizes = [500, 1000, 5000, 10000]
    
    generate_synthetic_data(target_sample_sizes)