"""
generate_diverse_data.py
========================
Generates diverse synthetic datasets (500, 1000, 5000 students) 
with injected pedagogical archetypes to properly test Knowledge Tracing models.
"""

import pandas as pd
import numpy as np
import random
import argparse
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(num_students):
    num_skills = 72
    interactions_per_student = 72 
    skill_diff = np.random.normal(0, 1.0, num_skills)
    data = []
    
    print(f"🧬 Generating Diverse Data for {num_students} Students...")
    for student_id in tqdm(range(num_students)):
        rand_val = random.random()
        
        # 1. High Achiever (20%)
        if rand_val < 0.20:
            theta, slip, guess = np.random.normal(1.5, 0.5), np.random.uniform(0.01, 0.05), np.random.uniform(0.01, 0.10)
            time_dist = (30, 10)
        # 2. Struggling (20%)
        elif rand_val < 0.40:
            theta, slip, guess = np.random.normal(-1.5, 0.5), np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)
            time_dist = (60, 20)
        # 3. Careless / High Slip (20%)
        elif rand_val < 0.60:
            theta, slip, guess = np.random.normal(1.0, 0.5), np.random.uniform(0.20, 0.40), np.random.uniform(0.05, 0.15)
            time_dist = (40, 15)
        # 4. Lucky Guesser (20%)
        elif rand_val < 0.80:
            theta, slip, guess = np.random.normal(-1.0, 0.5), np.random.uniform(0.05, 0.15), np.random.uniform(0.25, 0.45)
            time_dist = (25, 10)
        # 5. Average (20%)
        else:
            theta, slip, guess = np.random.normal(0, 0.5), np.random.uniform(0.05, 0.15), np.random.uniform(0.10, 0.20)
            time_dist = (45, 15)
            
        for _ in range(interactions_per_student):
            skill_id = random.randint(0, num_skills - 1)
            p_know = sigmoid(theta - skill_diff[skill_id])
            p_correct = p_know * (1 - slip) + (1 - p_know) * guess
            
            is_correct = 1 if random.random() < p_correct else 0
            time_taken = max(5, int(np.random.normal(*time_dist)))
                
            data.append({
                'student_id': student_id,
                'question_id': skill_id,
                'is_correct': is_correct,
                'response_time': time_taken
            })
            
    df = pd.DataFrame(data)
    output_name = f'synthetic_data_{num_students}_diverse.csv'
    df.to_csv(output_name, index=False)
    print(f"Created: {output_name} ({len(df)} interactions)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[500, 1000, 5000])
    args = parser.parse_args()
    
    for size in args.sizes:
        generate_data(size)