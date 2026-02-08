"""
Mock Data Generator for BKT Experiments.

Generates realistic student learning trajectories with various learner profiles.
This is used for testing and experimentation when real data is not available.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import random

from .schemas import (
    StudentInteraction, StudentSequence, Skill, Item, Dataset
)


class LearnerProfile:
    """Represents different types of learners with different characteristics."""
    
    FAST_LEARNER = {
        'name': 'Fast Learner',
        'p_init': 0.1,      # Low initial knowledge
        'p_learn': 0.30,    # High learning rate
        'p_guess': 0.15,    # Low guessing
        'p_slip': 0.05      # Low slip rate
    }
    
    SLOW_LEARNER = {
        'name': 'Slow Learner',
        'p_init': 0.1,      # Low initial knowledge
        'p_learn': 0.05,    # Low learning rate
        'p_guess': 0.25,    # Higher guessing
        'p_slip': 0.15      # Higher slip rate
    }
    
    PRIOR_KNOWLEDGE = {
        'name': 'Prior Knowledge',
        'p_init': 0.60,     # High initial knowledge
        'p_learn': 0.10,    # Medium learning rate
        'p_guess': 0.10,    # Low guessing
        'p_slip': 0.08      # Low slip rate
    }
    
    STRUGGLING = {
        'name': 'Struggling',
        'p_init': 0.05,     # Very low initial knowledge
        'p_learn': 0.08,    # Low learning rate
        'p_guess': 0.30,    # High guessing
        'p_slip': 0.20      # High slip rate
    }
    
    AVERAGE = {
        'name': 'Average',
        'p_init': 0.20,     # Medium initial knowledge
        'p_learn': 0.15,    # Medium learning rate
        'p_guess': 0.20,    # Medium guessing
        'p_slip': 0.10      # Medium slip rate
    }
    
    @classmethod
    def get_all_profiles(cls) -> List[Dict]:
        """Get all predefined learner profiles."""
        return [
            cls.FAST_LEARNER,
            cls.SLOW_LEARNER,
            cls.PRIOR_KNOWLEDGE,
            cls.STRUGGLING,
            cls.AVERAGE
        ]
    
    @classmethod
    def sample_profile(cls) -> Dict:
        """Randomly sample a learner profile."""
        return random.choice(cls.get_all_profiles())


class MockDataGenerator:
    """
    Generates synthetic student learning data for BKT experiments.
    
    The generator creates realistic learning trajectories by simulating
    the BKT process: students have a latent knowledge state that evolves
    over time, and their responses are generated based on this state.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the mock data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.seed = seed
    
    def generate_dataset(
        self,
        num_students: int = 100,
        num_skills: int = 5,
        num_items_per_skill: int = 20,
        min_attempts_per_student: int = 20,
        max_attempts_per_student: int = 100,
        skill_names: Optional[List[str]] = None,
        include_timestamps: bool = True,
        forgetting_rate: float = 0.0
    ) -> Dataset:
        """
        Generate a complete dataset with students, skills, and items.
        
        Args:
            num_students: Number of students to generate
            num_skills: Number of skills/knowledge components
            num_items_per_skill: Number of practice items per skill
            min_attempts_per_student: Minimum practice attempts per student
            max_attempts_per_student: Maximum practice attempts per student
            skill_names: Optional custom skill names (defaults to calculus topics)
            include_timestamps: Whether to include realistic timestamps
            forgetting_rate: Probability of forgetting (0 = no forgetting)
            
        Returns:
            Complete dataset with student sequences
        """
        # Generate skills
        skills = self._generate_skills(num_skills, skill_names)
        
        # Generate items
        items = self._generate_items(skills, num_items_per_skill)
        
        # Generate student sequences
        sequences = []
        for i in range(num_students):
            student_id = f"student_{i:04d}"
            num_attempts = random.randint(min_attempts_per_student, max_attempts_per_student)
            
            sequence = self._generate_student_sequence(
                student_id=student_id,
                skills=list(skills.values()),
                items=list(items.values()),
                num_attempts=num_attempts,
                include_timestamps=include_timestamps,
                forgetting_rate=forgetting_rate
            )
            sequences.append(sequence)
        
        # Create dataset
        dataset = Dataset(
            sequences=sequences,
            skills=skills,
            items=items
        )
        
        return dataset
    
    def _generate_skills(
        self, 
        num_skills: int,
        custom_names: Optional[List[str]] = None
    ) -> Dict[str, Skill]:
        """Generate skills/knowledge components."""
        default_calculus_topics = [
            "Limits",
            "Derivatives",
            "Chain Rule",
            "Integration",
            "U-Substitution",
            "Integration by Parts",
            "Applications of Derivatives",
            "Min/Max Problems",
            "Related Rates",
            "Area Under Curve"
        ]
        
        if custom_names:
            names = custom_names[:num_skills]
        else:
            names = default_calculus_topics[:num_skills]
        
        skills = {}
        for i in range(num_skills):
            skill_id = f"skill_{i:02d}"
            name = names[i] if i < len(names) else f"Skill {i+1}"
            
            # Add prerequisite relationships (simpler skills before complex ones)
            prerequisites = []
            if i > 0 and random.random() < 0.3:  # 30% chance of having prerequisite
                num_prereq = random.randint(1, min(2, i))
                prerequisites = [f"skill_{j:02d}" for j in 
                               random.sample(range(i), num_prereq)]
            
            skills[skill_id] = Skill(
                skill_id=skill_id,
                name=name,
                description=f"Knowledge component: {name}",
                prerequisite_skills=prerequisites
            )
        
        return skills
    
    def _generate_items(
        self,
        skills: Dict[str, Skill],
        num_items_per_skill: int
    ) -> Dict[str, Item]:
        """Generate practice items for each skill."""
        items = {}
        
        for skill_id, skill in skills.items():
            for i in range(num_items_per_skill):
                item_id = f"{skill_id}_item_{i:03d}"
                
                # Generate difficulty: distributed around mean with some variance
                # Difficulty affects slip/guess rates in contextu al BKT
                difficulty = np.random.normal(0.0, 0.5)  # Mean 0, std 0.5
                
                items[item_id] = Item(
                    item_id=item_id,
                    skill_id=skill_id,
                    difficulty=difficulty,
                    text=f"Practice problem for {skill.name}",
                    metadata={'difficulty_level': self._difficulty_to_level(difficulty)}
                )
        
        return items
    
    def _difficulty_to_level(self, difficulty: float) -> str:
        """Convert numeric difficulty to categorical level."""
        if difficulty < -0.5:
            return "easy"
        elif difficulty < 0.5:
            return "medium"
        else:
            return "hard"
    
    def _generate_student_sequence(
        self,
        student_id: str,
        skills: List[Skill],
        items: List[Item],
        num_attempts: int,
        include_timestamps: bool,
        forgetting_rate: float
    ) -> StudentSequence:
        """
        Generate a realistic learning sequence for one student.
        
        Uses the BKT generative process:
        1. Student starts with initial knowledge state for each skill
        2. For each attempt:
           - Select an item (skill)
           - Generate response based on knowledge state and slip/guess
           - Update knowledge state (learning or forgetting)
        """
        # Assign a learner profile to this student
        profile = LearnerProfile.sample_profile()
        
        # Initialize knowledge states for each skill
        knowledge_states = {}
        for skill in skills:
            knowledge_states[skill.skill_id] = (
                1.0 if np.random.random() < profile['p_init'] else 0.0
            )
        
        # Get parameters from profile
        p_learn = profile['p_learn']
        p_guess = profile['p_guess']
        p_slip = profile['p_slip']
        
        # Generate interactions
        interactions = []
        current_time = datetime.now() if include_timestamps else None
        
        # Track which skills student has practiced (for curriculum)
        practiced_skills = set()
        
        for attempt_num in range(num_attempts):
            # Select skill (curriculum model: practice all skills, but focus on struggling ones)
            if len(practiced_skills) < len(skills):
                # Early on: try all skills
                available_skills = [s for s in skills if s.skill_id not in practiced_skills]
                if available_skills:
                    skill = random.choice(available_skills)
                else:
                    skill = random.choice(skills)
            else:
                # Later: focus on skills not yet mastered
                unmastered = [s for s in skills if knowledge_states[s.skill_id] == 0.0]
                if unmastered and random.random() < 0.7:
                    skill = random.choice(unmastered)
                else:
                    skill = random.choice(skills)
            
            practiced_skills.add(skill.skill_id)
            
            # Select item from this skill
            skill_items = [item for item in items if item.skill_id == skill.skill_id]
            item = random.choice(skill_items)
            
            # Generate response based on knowledge state
            knows = knowledge_states[skill.skill_id] == 1.0
            
            if knows:
                # If knows: can slip
                correct = 0 if np.random.random() < p_slip else 1
            else:
                # If doesn't know: can guess
                correct = 1 if np.random.random() < p_guess else 0
            
            # Create interaction
            time_taken = np.random.exponential(60.0) + 10.0  # 10-300 seconds typically
            
            interaction = StudentInteraction(
                student_id=student_id,
                item_id=item.item_id,
                skill_id=skill.skill_id,
                correct=correct,
                timestamp=current_time,
                time_taken_seconds=time_taken
            )
            interactions.append(interaction)
            
            # Update knowledge state (learning or forgetting)
            if knowledge_states[skill.skill_id] == 0.0:
                # Opportunity to learn
                if np.random.random() < p_learn:
                    knowledge_states[skill.skill_id] = 1.0
            else:
                # Opportunity to forget (if enabled)
                if forgetting_rate > 0 and np.random.random() < forgetting_rate:
                    knowledge_states[skill.skill_id] = 0.0
            
            # Advance time
            if include_timestamps:
                # Time between attempts: 1 min to 3 days
                time_gap = np.random.exponential(3600) + 60  # seconds
                current_time += timedelta(seconds=time_gap)
        
        # Create sequence
        sequence = StudentSequence(
            student_id=student_id,
            interactions=interactions,
            metadata={
                'profile': profile['name'],
                'num_skills_practiced': len(practiced_skills),
                'final_accuracy': sum(i.correct for i in interactions) / len(interactions)
            }
        )
        
        return sequence
    
    def generate_simple_dataset(
        self,
        true_params: Dict[str, Dict[str, float]],
        num_students: int = 50,
        attempts_per_student: int = 30
    ) -> Dataset:
        """
        Generate dataset with known ground-truth parameters.
        Useful for testing parameter estimation algorithms.
        
        Args:
            true_params: Dictionary mapping skill_id to BKT parameters
                        e.g., {'skill_00': {'p_init': 0.2, 'p_learn': 0.15, ...}}
            num_students: Number of students
            attempts_per_student: Attempts per student per skill
            
        Returns:
            Dataset generated with known parameters
        """
        skills = {}
        items = {}
        
        # Create skills and items based on true_params
        for skill_id, params in true_params.items():
            skills[skill_id] = Skill(
                skill_id=skill_id,
                name=skill_id.replace('_', ' ').title()
            )
            
            # Create 10 items per skill
            for i in range(10):
                item_id = f"{skill_id}_item_{i:03d}"
                items[item_id] = Item(
                    item_id=item_id,
                    skill_id=skill_id
                )
        
        # Generate student sequences using known parameters
        sequences = []
        for student_idx in range(num_students):
            student_id = f"student_{student_idx:04d}"
            interactions = []
            
            # Track knowledge state for each skill
            knowledge_states = {}
            for skill_id, params in true_params.items():
                knowledge_states[skill_id] = (
                    1.0 if np.random.random() < params['p_init'] else 0.0
                )
            
            # Generate attempts for each skill
            for skill_id, params in true_params.items():
                skill_items = [item for item in items.values() if item.skill_id == skill_id]
                
                for _ in range(attempts_per_student):
                    item = random.choice(skill_items)
                    knows = knowledge_states[skill_id] == 1.0
                    
                    # Generate response
                    if knows:
                        correct = 0 if np.random.random() < params['p_slip'] else 1
                    else:
                        correct = 1 if np.random.random() < params['p_guess'] else 0
                    
                    interactions.append(StudentInteraction(
                        student_id=student_id,
                        item_id=item.item_id,
                        skill_id=skill_id,
                        correct=correct
                    ))
                    
                    # Update knowledge
                    if knowledge_states[skill_id] == 0.0:
                        if np.random.random() < params['p_learn']:
                            knowledge_states[skill_id] = 1.0
            
            # Shuffle interactions to simulate realistic practice order
            random.shuffle(interactions)
            
            sequences.append(StudentSequence(
                student_id=student_id,
                interactions=interactions
            ))
        
        return Dataset(
            sequences=sequences,
            skills=skills,
            items=items
        )
