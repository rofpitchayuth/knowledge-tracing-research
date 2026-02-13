from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class StudentInteraction:
    student_id: str
    item_id: str
    skill_id: str
    correct: int  # 0 or 1
    timestamp: Optional[datetime] = None
    time_taken_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.correct not in [0, 1]:
            raise ValueError(f"correct must be 0 or 1, got {self.correct}")


@dataclass
class StudentSequence:
    student_id: str
    interactions: List[StudentInteraction] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_interaction(self, interaction: StudentInteraction):
        if interaction.student_id != self.student_id:
            raise ValueError("Interaction student_id does not match sequence student_id")
        self.interactions.append(interaction)
    
    def get_skill_sequence(self, skill_id: str) -> List[StudentInteraction]:
        return [i for i in self.interactions if i.skill_id == skill_id]
    
    @property
    def num_interactions(self) -> int:
        return len(self.interactions)
    
    @property
    def accuracy(self) -> float:
        if not self.interactions:
            return 0.0
        return sum(i.correct for i in self.interactions) / len(self.interactions)


@dataclass
class Skill:
    skill_id: str
    name: str
    description: Optional[str] = None
    prerequisite_skills: List[str] = field(default_factory=list)


@dataclass
class Item:
    item_id: str
    skill_id: str
    difficulty: Optional[float] = None  # Item difficulty parameter
    text: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Dataset:
    sequences: List[StudentSequence] = field(default_factory=list)
    skills: Dict[str, Skill] = field(default_factory=dict)
    items: Dict[str, Item] = field(default_factory=dict)
    
    def add_sequence(self, sequence: StudentSequence):
        self.sequences.append(sequence)
    
    @property
    def num_students(self) -> int:
        return len(self.sequences)
    
    @property
    def num_interactions(self) -> int:
        return sum(seq.num_interactions for seq in self.sequences)
    
    def get_skill_ids(self) -> List[str]:
        return list(self.skills.keys())
