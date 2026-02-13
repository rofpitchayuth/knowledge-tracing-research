"""
Improved Bayesian Knowledge Tracing (IBKT) with Time.

This model extends Standard BKT by incorporating response time into the
Slip and Guess probabilities.

Hypothesis: 
- Fast responses are more likely to be Guesses (if correct) or Slips (if incorrect due to rushing).
- Slow responses might indicate careful thinking or confusion.

Model:
    P(G|T) = Sigmoid(w_g * log(T) + b_g)
    P(S|T) = Sigmoid(w_s * log(T) + b_s)
    
    P(Learn) and P(Init) remain standard parameters per skill.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.special import expit  # Sigmoid function
from scipy.optimize import minimize

from models.base import KnowledgeTracingModel
from data.schemas import Dataset, StudentInteraction, StudentSequence

class ImprovedBKT(KnowledgeTracingModel):
    """
    Improved BKT that uses response time to adjust Slip and Guess probabilities.
    """
    
    def __init__(self, use_log_time: bool = True):
        super().__init__()
        self.use_log_time = use_log_time
        self.params = {}  # skill_id -> {p_init, p_learn, w_s, b_s, w_g, b_g}
        self.mean_log_time = {} # skill_id -> mean log time (for normalization)
        self.std_log_time = {}  # skill_id -> std log time
        
    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Fit the Improved BKT model parameters using EM or Gradient Descent (Minimize NLL).
        Here we use L-BFGS-B minimization of Negative Log Likelihood.
        """
        # Group interactions by skill
        skill_interactions = {}
        skill_times = {}
        
        for seq in dataset.sequences:
            for interaction in seq.interactions:
                sid = interaction.skill_id
                if sid not in skill_interactions:
                    skill_interactions[sid] = []
                    skill_times[sid] = []
                
                skill_interactions[sid].append({
                    'student_id': seq.student_id,
                    'correct': interaction.correct,
                    'time': interaction.time_taken_seconds if interaction.time_taken_seconds else 60.0
                })
                skill_times[sid].append(interaction.time_taken_seconds if interaction.time_taken_seconds else 60.0)
                
        # Calculate time stats for normalization
        for skill_id, times in skill_times.items():
            filter_times = [t for t in times if t > 0]
            if not filter_times:
                self.mean_log_time[skill_id] = 0
                self.std_log_time[skill_id] = 1
            else:
                log_times = np.log(np.array(filter_times) + 1e-1) # Avoid log(0)
                self.mean_log_time[skill_id] = np.mean(log_times)
                self.std_log_time[skill_id] = np.std(log_times) + 1e-6

        # Optimize for each skill
        print(f"Fitting Improved BKT (Time) for {len(skill_interactions)} skills...")
        
        for skill_id, interactions in skill_interactions.items():
            # Organize by student for processing
            student_data = {}
            for intr in interactions:
                if intr['student_id'] not in student_data:
                    student_data[intr['student_id']] = []
                student_data[intr['student_id']].append(intr)
            
            # Initial params: p_init, p_learn, w_s, b_s, w_g, b_g
            # Init/Learn restricted to [0, 1] usually, but here we optimize raw logits or bounded
            # Let's optimize bounded for probs, and unbounded for weights
            
            # x = [p_init, p_learn, w_s, b_s, w_g, b_g]
            initial_guess = [0.5, 0.1, 0.0, -1.0, 0.0, -1.0] 
            bounds = [
                (0.001, 0.999), # p_init
                (0.001, 0.999), # p_learn
                (-5.0, 5.0),    # w_s
                (-5.0, 5.0),    # b_s
                (-5.0, 5.0),    # w_g
                (-5.0, 5.0)     # b_g
            ]
            
            res = minimize(
                self._negative_log_likelihood,
                initial_guess,
                args=(student_data, skill_id),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': kwargs.get('max_iterations', 100), 'disp': False}
            )
            
            self.params[skill_id] = {
                'p_init': res.x[0],
                'p_learn': res.x[1],
                'w_s': res.x[2],
                'b_s': res.x[3],
                'w_g': res.x[4],
                'b_g': res.x[5]
            }
            
        self.is_fitted = True

    def _negative_log_likelihood(self, params, student_data, skill_id):
        p_init, p_learn, w_s, b_s, w_g, b_g = params
        total_nll = 0.0
        
        # Precompute normalized time function
        mu = self.mean_log_time[skill_id]
        sigma = self.std_log_time[skill_id]
        
        for s_id, interactions in student_data.items():
            # Forward pass (HMM)
            # p(L_t) = prob student knows skill at step t
            
            p_known = p_init
            
            for intr in interactions:
                # Calculate dynamic slip/guess
                t = intr['time']
                norm_log_time = (np.log(t + 1e-1) - mu) / sigma
                
                p_slip = expit(w_s * norm_log_time + b_s)
                p_guess = expit(w_g * norm_log_time + b_g)
                
                # Likelihood of observation
                # P(Correct/L=1) = 1 - S
                # P(Correct/L=0) = G
                # P(Incorrect/L=1) = S
                # P(Incorrect/L=0) = 1 - G
                
                if intr['correct'] == 1:
                    p_obs_given_known = 1 - p_slip
                    p_obs_given_unknown = p_guess
                else:
                    p_obs_given_known = p_slip
                    p_obs_given_unknown = 1 - p_guess
                
                # Probability of this observation
                p_obs = p_known * p_obs_given_known + (1 - p_known) * p_obs_given_unknown
                
                # Clip to avoid log(0)
                p_obs = max(min(p_obs, 1.0 - 1e-9), 1e-9)
                total_nll -= np.log(p_obs)
                
                # Posterior update (Bayes rule)
                p_known_given_obs = (p_known * p_obs_given_known) / p_obs
                
                # Update for next step (Learning)
                p_known = p_known_given_obs + (1 - p_known_given_obs) * p_learn
                
        return total_nll

    def predict_next(self, student_id: str, history: List[StudentInteraction], next_skill: str) -> float:
        """Predict correctness probability for next item."""
        if next_skill not in self.params:
            return 0.5  # Default/Fallback
            
        params = self.params[next_skill]
        p_init = params['p_init']
        p_learn = params['p_learn']
        w_s, b_s = params['w_s'], params['b_s']
        w_g, b_g = params['w_g'], params['b_g']
        
        # Calculate current knowledge state P(L)
        p_known = p_init
        
        mu = self.mean_log_time.get(next_skill, 0)
        sigma = self.std_log_time.get(next_skill, 1)
        
        # Replay history for this skill
        skill_history = [i for i in history if i.skill_id == next_skill]
        
        for intr in skill_history:
            # Time of PARENT interaction
            t = intr.time_taken_seconds if intr.time_taken_seconds else 60.0
            norm_log_time = (np.log(t + 1e-1) - mu) / sigma
            
            p_slip = expit(w_s * norm_log_time + b_s)
            p_guess = expit(w_g * norm_log_time + b_g)
            
            if intr.correct == 1:
                p_obs_given_known = 1 - p_slip
                p_obs_given_unknown = p_guess
            else:
                p_obs_given_known = p_slip
                p_obs_given_unknown = 1 - p_guess
                
            p_obs = p_known * p_obs_given_known + (1 - p_known) * p_obs_given_unknown
            p_obs = max(1e-9, p_obs) # Safety
            
            p_known_given_obs = (p_known * p_obs_given_known) / p_obs
            p_known = p_known_given_obs + (1 - p_known_given_obs) * p_learn
            
        # For prediction, we don't know the NEXT time yet.
        # We assume "average" time for the prediction probability?
        # Or we ask the user to provide estimated time.
        # Standard KT prediction involves P(Correct) magnitude.
        # Since we don't have T_next, we use Mean T for prediction (Expected Value or Mode)
        # Using 0 (mean in normalized space)
        
        expected_norm_time = 0.0 # corresponds to geometric mean of time
        
        pred_slip = expit(w_s * expected_norm_time + b_s)
        pred_guess = expit(w_g * expected_norm_time + b_g)
        
        p_correct = p_known * (1 - pred_slip) + (1 - p_known) * pred_guess
        return p_correct

    def get_knowledge_state(self, student_id: str, skill_id: str, history: List[StudentInteraction]) -> float:
        # Re-use logic from predict_next to get p_known
        if skill_id not in self.params: return 0.0
        
        params = self.params[skill_id]
        p_known = params['p_init']
        p_learn = params['p_learn']
        w_s, b_s = params['w_s'], params['b_s']
        w_g, b_g = params['w_g'], params['b_g']
        mu = self.mean_log_time.get(skill_id, 0)
        sigma = self.std_log_time.get(skill_id, 1)

        skill_history = [i for i in history if i.skill_id == skill_id]
        
        for intr in skill_history:
            t = intr.time_taken_seconds if intr.time_taken_seconds else 60.0
            norm_log_time = (np.log(t + 1e-1) - mu) / sigma
            
            p_slip = expit(w_s * norm_log_time + b_s)
            p_guess = expit(w_g * norm_log_time + b_g)
            
            if intr.correct == 1:
                p_obs_given_known = 1 - p_slip
                p_obs_given_unknown = p_guess
            else:
                p_obs_given_known = p_slip
                p_obs_given_unknown = 1 - p_guess
                
            p_obs = p_known * p_obs_given_known + (1 - p_known) * p_obs_given_unknown
            p_obs = max(1e-9, p_obs) 
            
            p_known_given_obs = (p_known * p_obs_given_known) / p_obs
            p_known = p_known_given_obs + (1 - p_known_given_obs) * p_learn
            
        return p_known
