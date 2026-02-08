"""
Individualized Bayesian Knowledge Tracing (IBKT)

Extension of Standard BKT that allows per-student parameters, particularly
for P(L0) (initial knowledge) and P(T) (learning rate).

This addresses the limitation that standard BKT assumes all students learn
the same way. In reality, students have different:
- Prior knowledge levels (P(L0) varies)
- Learning rates (P(T) varies)
- Potentially different guess/slip tendencies

Based on research by:
- Pardos & Heffernan (2010) - Per-student parameters
- Yudelson et al. (2013) - Individualization in BKT

Key Insight: Some students are "fast learners" (high P(T)), others are
"slow learners" (low P(T)). One-size-fits-all parameters lose this nuance.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import copy

# Direct imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.base import ParametricBKTModel
from data.schemas import StudentInteraction, StudentSequence, Dataset


@dataclass
class IndividualizedBKTParameters:
    """
    Parameters for Individualized BKT.
    
    Structure:
    - Global parameters for slip/guess (shared across students)
    - Per-student parameters for init/learn (individual differences)
    """
    # Global parameters (same for all students)
    p_guess: float = 0.20    # P(G): Global guess rate
    p_slip: float = 0.10     # P(S): Global slip rate
    
    # Per-student parameters (will be Dict[student_id, float])
    student_p_init: Dict[str, float] = field(default_factory=dict)   # P(L0) per student
    student_p_learn: Dict[str, float] = field(default_factory=dict)  # P(T) per student
    
    # Global defaults for new students
    default_p_init: float = 0.20
    default_p_learn: float = 0.15
    
    def get_student_init(self, student_id: str) -> float:
        """Get P(L0) for a student, or default if not seen."""
        return self.student_p_init.get(student_id, self.default_p_init)
    
    def get_student_learn(self, student_id: str) -> float:
        """Get P(T) for a student, or default if not seen."""
        return self.student_p_learn.get(student_id, self.default_p_learn)
    
    def set_student_params(self, student_id: str, p_init: float, p_learn: float):
        """Set individual parameters for a student."""
        self.student_p_init[student_id] = np.clip(p_init, 0.0, 1.0)
        self.student_p_learn[student_id] = np.clip(p_learn, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            'p_guess': self.p_guess,
            'p_slip': self.p_slip,
            'default_p_init': self.default_p_init,
            'default_p_learn': self.default_p_learn,
            'num_students': len(self.student_p_init),
            # Statistics about student parameters
            'p_init_mean': np.mean(list(self.student_p_init.values())) if self.student_p_init else self.default_p_init,
            'p_init_std': np.std(list(self.student_p_init.values())) if self.student_p_init else 0.0,
            'p_learn_mean': np.mean(list(self.student_p_learn.values())) if self.student_p_learn else self.default_p_learn,
            'p_learn_std': np.std(list(self.student_p_learn.values())) if self.student_p_learn else 0.0,
        }


class IndividualizedBKT(ParametricBKTModel):
    """
    Individualized BKT with per-student parameters.
    
    Key Differences from Standard BKT:
    1. Each student has their own P(L0) and P(T)
    2. P(G) and P(S) are still global (shared)
    3. Can identify student groups (fast/slow learners)
    4. Better fits heterogeneous student populations
    
    Trade-offs:
    + More accurate for diverse students
    + Can identify struggling students early
    - More parameters to estimate (2 * num_students)
    - Requires more data per student
    - Cold start problem for new students
    """
    
    def __init__(self, skills: Optional[List[str]] = None):
        """
        Initialize Individualized BKT model.
        
        Args:
            skills: List of skill IDs to initialize
        """
        super().__init__()
        self.model_name = "Individualized BKT"
        self.skills_params: Dict[str, IndividualizedBKTParameters] = {}
        self.knowledge_states: Dict[Tuple[str, str], float] = {}
        
        if skills:
            for skill in skills:
                self.skills_params[skill] = IndividualizedBKTParameters()
    
    def _ensure_skill_exists(self, skill_id: str) -> None:
        """Initialize skill parameters if they don't exist."""
        if skill_id not in self.skills_params:
            self.skills_params[skill_id] = IndividualizedBKTParameters()
    
    def _clip_probability(self, p: float) -> float:
        """Clip probability to valid range [0, 1]."""
        return np.clip(p, 0.0, 1.0)
    
    def get_parameters(self, skill_id: str) -> Dict[str, float]:
        """Get global parameters for a skill."""
        self._ensure_skill_exists(skill_id)
        return self.skills_params[skill_id].to_dict()
    
    def set_parameters(self, skill_id: str, params: Dict[str, float]) -> None:
        """Set global parameters (P(G), P(S)) for a skill."""
        self._ensure_skill_exists(skill_id)
        if 'p_guess' in params:
            self.skills_params[skill_id].p_guess = params['p_guess']
        if 'p_slip' in params:
            self.skills_params[skill_id].p_slip = params['p_slip']
        if 'default_p_init' in params:
            self.skills_params[skill_id].default_p_init = params['default_p_init']
        if 'default_p_learn' in params:
            self.skills_params[skill_id].default_p_learn = params['default_p_learn']
    
    def get_student_parameters(self, skill_id: str, student_id: str) -> Dict[str, float]:
        """Get all parameters (global + individual) for a specific student."""
        self._ensure_skill_exists(skill_id)
        params_obj = self.skills_params[skill_id]
        
        return {
            'p_init': params_obj.get_student_init(student_id),
            'p_learn': params_obj.get_student_learn(student_id),
            'p_guess': params_obj.p_guess,
            'p_slip': params_obj.p_slip
        }
    
    def get_knowledge_state(self, student_id: str, skill_id: str,
                           history: List[StudentInteraction]) -> float:
        """
        Get current knowledge state for a student-skill.
        
        Uses student-specific P(L0) and P(T).
        """
        self._ensure_skill_exists(skill_id)
        params_obj = self.skills_params[skill_id]
        
        # Filter history for this skill
        skill_history = [h for h in history if h.skill_id == skill_id]
        
        # Start with student's individual P(L0)
        p_know = params_obj.get_student_init(student_id)
        
        # Update through each interaction using student's P(T)
        p_learn = params_obj.get_student_learn(student_id)
        
        for interaction in skill_history:
            p_know = self._update_single(
                p_know, interaction.correct,
                p_learn, params_obj.p_guess, params_obj.p_slip
            )
        
        return p_know
    
    def predict_next(self, student_id: str, history: List[StudentInteraction],
                     next_skill: str) -> float:
        """
        Predict probability of answering next question correctly.
        
        Uses student-specific parameters.
        """
        self._ensure_skill_exists(next_skill)
        params_obj = self.skills_params[next_skill]
        
        # Get current knowledge state (uses student's params)
        p_know = self.get_knowledge_state(student_id, next_skill, history)
        
        # P(Correct) = P(L)*(1-S) + (1-P(L))*G
        p_correct = p_know * (1.0 - params_obj.p_slip) + (1.0 - p_know) * params_obj.p_guess
        
        return self._clip_probability(p_correct)
    
    def _update_single(self, p_know_before: float, correct: int,
                      p_learn: float, p_guess: float, p_slip: float) -> float:
        """
        Single BKT update step.
        
        Note: This uses student-specific p_learn.
        """
        # Evidence probabilities
        p_correct_if_know = 1.0 - p_slip
        p_correct_if_not_know = p_guess
        
        # Bayes' theorem
        if correct == 1:
            numerator = p_know_before * p_correct_if_know
            denominator = (p_know_before * p_correct_if_know +
                          (1.0 - p_know_before) * p_correct_if_not_know)
        else:
            numerator = p_know_before * (1.0 - p_correct_if_know)
            denominator = (p_know_before * (1.0 - p_correct_if_know) +
                          (1.0 - p_know_before) * (1.0 - p_correct_if_not_know))
        
        if denominator < 1e-10:
            p_know_after_evidence = p_know_before
        else:
            p_know_after_evidence = self._clip_probability(numerator / denominator)
        
        # Transition (using student-specific P(T))
        p_know_after_transition = p_know_after_evidence + (1.0 - p_know_after_evidence) * p_learn
        
        return self._clip_probability(p_know_after_transition)
    
    def fit(self, dataset: Dataset, max_iterations: int = 100,
            tolerance: float = 1e-4, verbose: bool = False) -> None:
        """
        Fit Individualized BKT using EM algorithm.
        
        E-step: Compute expected knowledge states per student
        M-step: Update global params (G, S) and individual params (L0, T) per student
        
        Args:
            dataset: Training dataset
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance
            verbose: Print training progress
        """
        if verbose:
            print(f"Fitting {self.model_name} on {dataset.num_students} students...")
            print(f"Estimating {dataset.num_students * 2} individual parameters + 2 global parameters per skill")
        
        # Initialize parameters for all skills
        for skill_id in dataset.get_skill_ids():
            self._ensure_skill_exists(skill_id)
            
            # Initialize individual parameters for all students
            for sequence in dataset.sequences:
                student_id = sequence.student_id
                params_obj = self.skills_params[skill_id]
                
                # Start with defaults
                if student_id not in params_obj.student_p_init:
                    params_obj.student_p_init[student_id] = params_obj.default_p_init
                if student_id not in params_obj.student_p_learn:
                    params_obj.student_p_learn[student_id] = params_obj.default_p_learn
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step
            expected_states = self._e_step(dataset)
            
            # M-step
            self._m_step(dataset, expected_states)
            
            # Log-likelihood
            log_likelihood = self._compute_log_likelihood(dataset)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}")
            
            # Convergence check
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_log_likelihood = log_likelihood
        
        self.is_fitted = True
        
        if verbose:
            print(f"Training completed. Final log-likelihood: {log_likelihood:.4f}")
            
            # Print parameter statistics
            for skill_id in sorted(self.skills_params.keys()):
                params_dict = self.get_parameters(skill_id)
                print(f"\n{skill_id}:")
                print(f"  Global: G={params_dict['p_guess']:.3f}, S={params_dict['p_slip']:.3f}")
                print(f"  P(L0): mean={params_dict['p_init_mean']:.3f}, std={params_dict['p_init_std']:.3f}")
                print(f"  P(T):  mean={params_dict['p_learn_mean']:.3f}, std={params_dict['p_learn_std']:.3f}")
                print(f"  Students: {params_dict['num_students']}")
    
    def _e_step(self, dataset: Dataset) -> Dict[str, Dict[str, List[float]]]:
        """
        E-step: Compute expected knowledge states for each student.
        
        Returns nested dict: {skill_id: {student_id: [p_know_trajectory]}}
        """
        expected_states = {}
        
        for skill_id in self.skills_params.keys():
            expected_states[skill_id] = {}
        
        for sequence in dataset.sequences:
            student_id = sequence.student_id
            
            # Group by skill
            skill_sequences = {}
            for interaction in sequence.interactions:
                if interaction.skill_id not in skill_sequences:
                    skill_sequences[interaction.skill_id] = []
                skill_sequences[interaction.skill_id].append(interaction)
            
            # Compute trajectory for each skill
            for skill_id, interactions in skill_sequences.items():
                if skill_id not in self.skills_params:
                    continue
                
                params_obj = self.skills_params[skill_id]
                p_init = params_obj.get_student_init(student_id)
                p_learn = params_obj.get_student_learn(student_id)
                
                trajectory = []
                p_know = p_init
                
                for interaction in interactions:
                    trajectory.append(p_know)
                    p_know = self._update_single(
                        p_know, interaction.correct,
                        p_learn, params_obj.p_guess, params_obj.p_slip
                    )
                
                expected_states[skill_id][student_id] = trajectory
        
        return expected_states
    
    def _m_step(self, dataset: Dataset, expected_states: Dict[str, Dict[str, List[float]]]) -> None:
        """
        M-step: Update parameters based on expected states.
        
        Updates:
        1. Global P(G) and P(S)
        2. Individual P(L0) and P(T) per student
        """
        for skill_id in self.skills_params.keys():
            params_obj = self.skills_params[skill_id]
            
            # Accumulators for global params
            total_guesses = 0.0
            total_guess_opportunities = 0.0
            total_slips = 0.0
            total_slip_opportunities = 0.0
            
            # Get all interactions for this skill
            skill_data = {}  # student_id -> interactions
            for sequence in dataset.sequences:
                skill_interactions = [i for i in sequence.interactions if i.skill_id == skill_id]
                if skill_interactions:
                    skill_data[sequence.student_id] = skill_interactions
            
            # Update per-student parameters
            for student_id, interactions in skill_data.items():
                if student_id not in expected_states[skill_id]:
                    continue
                
                trajectory = expected_states[skill_id][student_id]
                
                if len(trajectory) == 0:
                    continue
                
                # Update P(L0) for this student - use first state
                new_p_init = trajectory[0]
                params_obj.student_p_init[student_id] = self._clip_probability(new_p_init)
                
                # Update P(T) for this student - estimate from transitions
                # Simplified: use average learn rate needed
                learn_count = 0.0
                learn_opportunities = 0.0
                
                for i, interaction in enumerate(interactions):
                    if i >= len(trajectory):
                        break
                    
                    p_know = trajectory[i]
                    p_not_know = 1.0 - p_know
                    
                    # Count for guess/slip (global)
                    if interaction.correct == 1:
                        total_guesses += p_not_know
                        total_slip_opportunities += p_know
                    else:
                        total_slips += p_know
                        total_guess_opportunities += p_not_know
                    
                    total_guess_opportunities += p_not_know
                    total_slip_opportunities += p_know
                    
                    # Estimate learning (simplified)
                    if p_not_know > 0.1:  # Only when there's room to learn
                        learn_opportunities += p_not_know
                
                # Update student's P(T)
                if learn_opportunities > 0:
                    # Simple heuristic: higher performing students → higher P(T)
                    accuracy = sum(i.correct for i in interactions) / len(interactions)
                    new_p_learn = min(0.5, accuracy * 0.3)  # Scale based on performance
                    params_obj.student_p_learn[student_id] = self._clip_probability(new_p_learn)
            
            # Update global P(G) and P(S)
            if total_guess_opportunities > 0:
                params_obj.p_guess = self._clip_probability(total_guesses / total_guess_opportunities)
            
            if total_slip_opportunities > 0:
                params_obj.p_slip = self._clip_probability(total_slips / total_slip_opportunities)
    
    def _compute_log_likelihood(self, dataset: Dataset) -> float:
        """Compute log-likelihood of data given current parameters."""
        log_likelihood = 0.0
        
        for sequence in dataset.sequences:
            history = []
            for interaction in sequence.interactions:
                p_correct = self.predict_next(sequence.student_id, history, interaction.skill_id)
                
                if interaction.correct == 1:
                    log_likelihood += np.log(max(p_correct, 1e-10))
                else:
                    log_likelihood += np.log(max(1.0 - p_correct, 1e-10))
                
                history.append(interaction)
        
        return log_likelihood
    
    def get_student_profile(self, student_id: str, skill_id: str) -> Dict[str, str]:
        """
        Classify student's learning profile based on parameters.
        
        Returns classification like "Fast Learner", "Struggling", etc.
        """
        params = self.get_student_parameters(skill_id, student_id)
        
        p_init = params['p_init']
        p_learn = params['p_learn']
        
        if p_init > 0.5:
            return {"profile": "Prior Knowledge", "description": "Already knows the skill"}
        elif p_learn > 0.25:
            return {"profile": "Fast Learner", "description": "Learns quickly from practice"}
        elif p_learn < 0.10:
            return {"profile": "Slow Learner", "description": "Needs more practice"}
        elif p_init < 0.15 and p_learn < 0.12:
            return {"profile": "Struggling", "description": "Low prior knowledge and slow learning"}
        else:
            return {"profile": "Average", "description": "Typical learning pattern"}
