from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime
from pathlib import Path

from .schemas import Dataset, StudentSequence, StudentInteraction, Skill, Item

class DataLoader:
    """
    Loads data from CSV files into the Dataset schema.
    Automatically maps arbitrary item_ids/skill_ids to continuous integers (0, 1, 2...)
    to prevent 'index out of range' errors in Deep Learning models.
    """
    
    @staticmethod
    def load_from_csv(
        filepath: str,
        col_student: str = 'student_id',
        col_skill: str = 'question_id',
        col_item: str = 'question_id',
        col_correct: str = 'is_correct',
        col_time: Optional[str] = None,
        col_time_taken: Optional[str] = 'response_time',
        time_format: str = '%Y-%m-%d %H:%M:%S'
    ) -> Dataset:
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        
        # Validate columns
        required = [col_student, col_item, col_correct]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        actual_col_skill = col_skill if col_skill in df.columns else col_item
        
        # ==========================================
        # THE FIX: Create Continuous ID Mapping
        # ==========================================
        unique_skills = df[actual_col_skill].unique()
        unique_items = df[col_item].unique()
        
        # Map original ID -> Continuous String ID ("0", "1", "2", ...)
        skill_id_map = {orig: str(idx) for idx, orig in enumerate(unique_skills)}
        item_id_map = {orig: str(idx) for idx, orig in enumerate(unique_items)}
        
        # Create Skills and Items dicts using mapped IDs
        skills: Dict[str, Skill] = {}
        for orig_sid in unique_skills:
            mapped_sid = skill_id_map[orig_sid]
            # Keep the original ID in the 'name' field for reference
            skills[mapped_sid] = Skill(skill_id=mapped_sid, name=str(orig_sid))
            
        items: Dict[str, Item] = {}
        for orig_iid in unique_items:
            mapped_iid = item_id_map[orig_iid]
            orig_skill = df[df[col_item] == orig_iid][actual_col_skill].iloc[0]
            mapped_skill = skill_id_map[orig_skill]
            items[mapped_iid] = Item(item_id=mapped_iid, skill_id=mapped_skill)
            
        # Create Student Sequences
        sequences: List[StudentSequence] = []
        
        # Group by student
        for student_id, group in df.groupby(col_student):
            if col_time and col_time in df.columns:
                try:
                    group[col_time] = pd.to_datetime(group[col_time])
                    group = group.sort_values(col_time)
                except Exception as e:
                    print(f"Warning: Could not parse timestamps for student {student_id}: {e}")
            
            interactions = []
            for _, row in group.iterrows():
                # Parse correctness
                raw_correct = row[col_correct]
                if isinstance(raw_correct, bool):
                    correct_val = 1 if raw_correct else 0
                elif str(raw_correct).lower() in ['true', 't', '1', '1.0']:
                    correct_val = 1
                elif str(raw_correct).lower() in ['false', 'f', '0', '0.0']:
                    correct_val = 0
                else:
                    continue
                
                timestamp = None
                if col_time and col_time in df.columns:
                    timestamp = row[col_time]
                    
                time_taken = None
                if col_time_taken and col_time_taken in df.columns:
                    try:
                        time_taken = float(row[col_time_taken])
                    except:
                        pass
                
                # Use the mapped contiguous IDs here!
                interactions.append(StudentInteraction(
                    student_id=str(student_id),
                    item_id=item_id_map[row[col_item]],
                    skill_id=skill_id_map[row[actual_col_skill]],
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