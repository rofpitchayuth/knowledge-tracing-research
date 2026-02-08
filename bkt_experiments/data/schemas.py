"""
Data schemas for BKT experiments.

This module defines data structures for student interactions, skills, and items.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class StudentInteraction:
    """Represents a single student interaction with a practice item."""
    student_id: str
    item_id: str
    skill_id: str
    correct: int  # 0 or 1
    timestamp: Optional[datetime] = None
    time_taken_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.correct not in [0, 1]:
            raise ValueError(f"correct must be 0 or 1, got {self.correct}")


@dataclass
class StudentSequence:
    """Represents a sequence of interactions for a single student."""
    student_id: str
    interactions: List[StudentInteraction] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_interaction(self, interaction: StudentInteraction):
        """Add an interaction to the sequence."""
        if interaction.student_id != self.student_id:
            raise ValueError("Interaction student_id does not match sequence student_id")
        self.interactions.append(interaction)
    
    def get_skill_sequence(self, skill_id: str) -> List[StudentInteraction]:
        """Get all interactions for a specific skill."""
        return [i for i in self.interactions if i.skill_id == skill_id]
    
    @property
    def num_interactions(self) -> int:
        """Total number of interactions."""
        return len(self.interactions)
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy across all interactions."""
        if not self.interactions:
            return 0.0
        return sum(i.correct for i in self.interactions) / len(self.interactions)


@dataclass
class Skill:
    """Represents a knowledge component/skill."""
    skill_id: str
    name: str
    description: Optional[str] = None
    prerequisite_skills: List[str] = field(default_factory=list)


@dataclass
class Item:
    """Represents a practice item/question."""
    item_id: str
    skill_id: str
    difficulty: Optional[float] = None  # Item difficulty parameter
    text: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Dataset:
    """Complete dataset for BKT experiments."""
    sequences: List[StudentSequence] = field(default_factory=list)
    skills: Dict[str, Skill] = field(default_factory=dict)
    items: Dict[str, Item] = field(default_factory=dict)
    
    def add_sequence(self, sequence: StudentSequence):
        """Add a student sequence to the dataset."""
        self.sequences.append(sequence)
    
    @property
    def num_students(self) -> int:
        """Total number of students."""
        return len(self.sequences)
    
    @property
    def num_interactions(self) -> int:
        """Total number of interactions across all students."""
        return sum(seq.num_interactions for seq in self.sequences)
    
    def get_skill_ids(self) -> List[str]:
        """Get all unique skill IDs in the dataset."""
        return list(self.skills.keys())
