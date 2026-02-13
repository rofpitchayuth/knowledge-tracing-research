from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime
from pathlib import Path

from .schemas import Dataset, StudentSequence, StudentInteraction, Skill, Item

class DataLoader:
    """
    Loads data from CSV files into the Dataset schema.
    """
    
    @staticmethod
    def load_from_csv(
        filepath: str,
        col_student: str = 'student_id',
        col_skill: str = 'skill_id',
        col_item: str = 'item_id',
        col_correct: str = 'correct',
        col_time: Optional[str] = 'timestamp',
        col_time_taken: Optional[str] = 'time_taken',
        time_format: str = '%Y-%m-%d %H:%M:%S'
    ) -> Dataset:
        """
        Load a dataset from a CSV file.
        
        Args:
            filepath: Path to CSV file
            col_student: Column name for student ID
            col_skill: Column name for skill/KC ID
            col_item: Column name for item/problem ID
            col_correct: Column name for correctness (0/1)
            col_time: Column name for timestamp (optional)
            col_time_taken: Column name for response time in seconds (optional)
            time_format: Format string for parsing timestamps
            
        Returns:
            Dataset object populated with the data
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        
        # Validate columns
        required = [col_student, col_skill, col_item, col_correct]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create Skills and Items dicts
        skills: Dict[str, Skill] = {}
        items: Dict[str, Item] = {}
        
        unique_skills = df[col_skill].unique()
        for sid in unique_skills:
            skills[str(sid)] = Skill(skill_id=str(sid), name=str(sid))
            
        unique_items = df[col_item].unique()
        for iid in unique_items:
            # Try to find skill for this item (assuming 1-to-1 mapping in data)
            # Take first skill associated with item
            item_skill = df[df[col_item] == iid][col_skill].iloc[0]
            items[str(iid)] = Item(item_id=str(iid), skill_id=str(item_skill))
            
        # Create Student Sequences
        sequences: List[StudentSequence] = []
        
        # Group by student
        for student_id, group in df.groupby(col_student):
            # Sort by time if available
            if col_time and col_time in df.columns:
                try:
                    group[col_time] = pd.to_datetime(group[col_time])
                    group = group.sort_values(col_time)
                except Exception as e:
                    print(f"Warning: Could not parse timestamps for student {student_id}: {e}")
            
            interactions = []
            for _, row in group.iterrows():
                # specific checks
                correct_val = int(row[col_correct])
                if correct_val not in [0, 1]:
                    continue # Skip invalid rows
                
                timestamp = None
                if col_time and col_time in df.columns:
                    timestamp = row[col_time]
                    
                time_taken = None
                if col_time_taken and col_time_taken in df.columns:
                    try:
                        time_taken = float(row[col_time_taken])
                    except:
                        pass
                
                # If no time taken, ImprovedBKT might default to 60s
                
                interactions.append(StudentInteraction(
                    student_id=str(student_id),
                    item_id=str(row[col_item]),
                    skill_id=str(row[col_skill]),
                    correct=correct_val,
                    timestamp=timestamp,
                    time_taken_seconds=time_taken
                ))
            
            if interactions:
                sequences.append(StudentSequence(
                    student_id=str(student_id),
                    interactions=interactions
                ))
        
        dataset = Dataset(sequences=sequences, skills=skills, items=items)
        print(f"Created dataset with {dataset.num_students} students and {dataset.num_interactions} interactions.")
        return dataset
