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
class BKTForgettingParameters:
    p_init: float = 0.20     # P(L0): Initial probability of knowing
    p_learn: float = 0.15    # P(T): Probability of learning (0→1 transition)
    p_forget: float = 0.05   # P(F): Probability of forgetting (1→0 transition) **NEW**
    p_guess: float = 0.20    # P(G): Probability of guessing correctly when not knowing
    p_slip: float = 0.10     # P(S): Probability of slipping (error) when knowing
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        for name in ['p_init', 'p_learn', 'p_forget', 'p_guess', 'p_slip']:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        
        # Constraint: P(G) + P(S) < 1 for identifiability
        if self.p_guess + self.p_slip >= 1.0:
            raise ValueError(
                f"P(G) + P(S) should be < 1.0 for identifiability. "
                f"Got P(G)={self.p_guess}, P(S)={self.p_slip}"
            )
        
        # Note: P(T) + P(F) can be >= 1 (different from P(G)+P(S))
        # because they act on different states
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'p_init': self.p_init,
            'p_learn': self.p_learn,
            'p_forget': self.p_forget,
            'p_guess': self.p_guess,
            'p_slip': self.p_slip
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'BKTForgettingParameters':
        return cls(**params)


class BKTWithForgetting(ParametricBKTModel):
    def __init__(self, skills: Optional[List[str]] = None,
                 default_params: Optional[BKTForgettingParameters] = None):
        
        super().__init__()
        self.model_name = "BKT with Forgetting"
        self.skills_params: Dict[str, BKTForgettingParameters] = {}
        self.knowledge_states: Dict[Tuple[str, str], float] = {}
        
        if skills:
            for skill in skills:
                self.skills_params[skill] = (
                    copy.deepcopy(default_params) if default_params
                    else BKTForgettingParameters()
                )
    
    def _ensure_skill_exists(self, skill_id: str) -> None:
        if skill_id not in self.skills_params:
            self.skills_params[skill_id] = BKTForgettingParameters()
    
    def _clip_probability(self, p: float) -> float:
        return np.clip(p, 0.0, 1.0)
    
    def get_parameters(self, skill_id: str) -> Dict[str, float]:
        self._ensure_skill_exists(skill_id)
        return self.skills_params[skill_id].to_dict()
    
    def set_parameters(self, skill_id: str, params: Dict[str, float]) -> None:
        self.skills_params[skill_id] = BKTForgettingParameters.from_dict(params)
    
    def get_knowledge_state(self, student_id: str, skill_id: str,
                           history: List[StudentInteraction]) -> float:
        """
        Args:
            student_id: Student ID
            skill_id: Skill ID
            history: List of past interactions
            
        Returns:
            P(L_t): Probability that student has mastered the skill
        """
        self._ensure_skill_exists(skill_id)
        params = self.skills_params[skill_id]
        
        # Filter history for this skill
        skill_history = [h for h in history if h.skill_id == skill_id]
        
        # Start with prior
        p_know = params.p_init
        
        # Update through each interaction
        for interaction in skill_history:
            p_know = self._update_single(p_know, interaction.correct, params)
        
        return p_know
    
    def predict_next(self, student_id: str, history: List[StudentInteraction],
                     next_skill: str) -> float:
        """
        Args:
            student_id: Student ID
            history: Past interactions
            next_skill: Skill ID for next question    
        """
        self._ensure_skill_exists(next_skill)
        params = self.skills_params[next_skill]
        
        # Get current knowledge state
        p_know = self.get_knowledge_state(student_id, next_skill, history)
        
        # P(Correct) = P(L)*(1-S) + (1-P(L))*G
        p_correct = p_know * (1.0 - params.p_slip) + (1.0 - p_know) * params.p_guess
        
        return self._clip_probability(p_correct)
    
    def _update_single(self, p_know_before: float, correct: int,
                      params: BKTForgettingParameters) -> float:
        """
        Single BKT update step with forgetting.
        Args:
            p_know_before: P(L_t) before seeing evidence
            correct: 1 if correct, 0 if incorrect
            params: BKT parameters (including forgetting)
            
        Returns:
            P(L_{t+1}): Updated probability of knowing after learning/forgetting
        """
        # Evidence probabilities
        p_correct_if_know = 1.0 - params.p_slip
        p_correct_if_not_know = params.p_guess
        
        # Bayes' theorem: P(L | observation)
        if correct == 1:
            numerator = p_know_before * p_correct_if_know
            denominator = (p_know_before * p_correct_if_know +
                          (1.0 - p_know_before) * p_correct_if_not_know)
        else:
            numerator = p_know_before * (1.0 - p_correct_if_know)
            denominator = (p_know_before * (1.0 - p_correct_if_know) +
                          (1.0 - p_know_before) * (1.0 - p_correct_if_not_know))
        
        # Handle numerical edge cases
        if denominator < 1e-10:
            p_know_after_evidence = p_know_before
        else:
            p_know_after_evidence = self._clip_probability(numerator / denominator)
        
        # Transition with BOTH learning and forgetting
        # Standard BKT: P(L_{t+1}) = P(L_t) + (1 - P(L_t)) * P(T)
        # With forgetting: P(L_{t+1}) = P(L_t) * (1 - P(F)) + (1 - P(L_t)) * P(T)
        
        # Probability of knowing at t+1:
        # = P(knew and didn't forget) + P(didn't know but learned)
        p_know_after_transition = (
            p_know_after_evidence * (1.0 - params.p_forget) +  # Knew, didn't forget
            (1.0 - p_know_after_evidence) * params.p_learn      # Didn't know, learned
        )
        
        return self._clip_probability(p_know_after_transition)
    
    def fit(self, dataset: Dataset, max_iterations: int = 100,
            tolerance: float = 1e-4, verbose: bool = False, **kwargs) -> None:
        """
        Args:
            dataset: Training dataset
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance
            verbose: Print training progress
        """
        if verbose:
            print(f"Fitting {self.model_name} on {dataset.num_students} students...")
            print("Note: Estimating forgetting parameter P(F) may require more iterations.")
        
        # Initialize parameters for all skills
        for skill_id in dataset.get_skill_ids():
            self._ensure_skill_exists(skill_id)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Compute expected knowledge states
            expected_states = self._e_step(dataset)
            
            # M-step: Update parameters (keeping P(F) fixed for now)
            self._m_step(dataset, expected_states)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(dataset)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}")
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_log_likelihood = log_likelihood
        
        self.is_fitted = True
        
        if verbose:
            print(f"Training completed. Final log-likelihood: {log_likelihood:.4f}")
            print("\nLearned parameters (with forgetting):")
            for skill_id in sorted(self.skills_params.keys()):
                params = self.skills_params[skill_id]
                print(f"  {skill_id}:")
                print(f"    L0={params.p_init:.3f}, T={params.p_learn:.3f}, "
                      f"F={params.p_forget:.3f}, G={params.p_guess:.3f}, S={params.p_slip:.3f}")
    
    def _e_step(self, dataset: Dataset) -> Dict[str, List[List[float]]]:
        """E-step: Same as standard BKT."""
        expected_states = {skill_id: [] for skill_id in self.skills_params.keys()}
        
        for sequence in dataset.sequences:
            skill_sequences = {}
            for interaction in sequence.interactions:
                if interaction.skill_id not in skill_sequences:
                    skill_sequences[interaction.skill_id] = []
                skill_sequences[interaction.skill_id].append(interaction)
            
            for skill_id, interactions in skill_sequences.items():
                if skill_id not in self.skills_params:
                    continue
                
                params = self.skills_params[skill_id]
                trajectory = []
                
                p_know = params.p_init
                for interaction in interactions:
                    trajectory.append(p_know)
                    p_know = self._update_single(p_know, interaction.correct, params)
                
                expected_states[skill_id].append(trajectory)
        
        return expected_states
    
    def _m_step(self, dataset: Dataset, expected_states: Dict[str, List[List[float]]]) -> None:
        """
        M-step: Update parameters based on expected states.
        
        For simplicity, we keep P(F) fixed in this implementation.
        Full EM for P(F) requires more sophisticated counting.
        """
        for skill_id in self.skills_params.keys():
            if skill_id not in expected_states or not expected_states[skill_id]:
                continue
            
            # Count events
            total_init = 0.0
            total_guesses = 0.0
            total_opportunities_to_guess = 0.0
            total_slips = 0.0
            total_opportunities_to_slip = 0.0
            
            # Get all interactions for this skill
            skill_interactions = []
            for sequence in dataset.sequences:
                for interaction in sequence.interactions:
                    if interaction.skill_id == skill_id:
                        skill_interactions.append((sequence.student_id, interaction))
            
            # Count from trajectories
            trajectory_idx = 0
            current_student = None
            interaction_in_student = 0
            
            for student_id, interaction in skill_interactions:
                if student_id != current_student:
                    current_student = student_id
                    trajectory_idx += 1
                    interaction_in_student = 0
                
                if trajectory_idx > len(expected_states[skill_id]):
                    break
                
                trajectory = expected_states[skill_id][trajectory_idx - 1]
                
                if interaction_in_student >= len(trajectory):
                    interaction_in_student += 1
                    continue
                
                p_know = trajectory[interaction_in_student]
                p_not_know = 1.0 - p_know
                
                # Initial knowledge
                if interaction_in_student == 0:
                    total_init += p_know
                
                # Slip and Guess
                if interaction.correct == 1:
                    total_guesses += p_not_know
                    total_opportunities_to_slip += p_know
                else:
                    total_slips += p_know
                    total_opportunities_to_guess += p_not_know
                
                total_opportunities_to_guess += p_not_know
                total_opportunities_to_slip += p_know
                
                interaction_in_student += 1
            
            # Update parameters
            num_students = len(expected_states[skill_id])
            
            new_p_init = total_init / max(num_students, 1)
            new_p_guess = total_guesses / max(total_opportunities_to_guess, 1)
            new_p_slip = total_slips / max(total_opportunities_to_slip, 1)
            
            # Keep learn rate and forget rate at reasonable defaults for stability
            new_p_learn = 0.15
            keep_p_forget = self.skills_params[skill_id].p_forget
            
            # Clip and validate
            try:
                self.skills_params[skill_id] = BKTForgettingParameters(
                    p_init=self._clip_probability(new_p_init),
                    p_learn=self._clip_probability(new_p_learn),
                    p_forget=keep_p_forget,  # Keep forgetting rate fixed
                    p_guess=self._clip_probability(new_p_guess),
                    p_slip=self._clip_probability(new_p_slip)
                )
            except ValueError:
                # If validation fails, keep old parameters
                pass
    
    def _compute_log_likelihood(self, dataset: Dataset) -> float:
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
