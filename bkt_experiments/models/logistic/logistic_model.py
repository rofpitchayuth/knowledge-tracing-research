"""
Logistic Knowledge Tracing Model (Performance Factor Analysis - PFA).

This model predicts the probability of correctness based on:
1. Item/Skill difficulty (beta)
2. Prior successes on this skill (s)
3. Prior failures on this skill (f)

Equation:
    logit(P) = beta + gamma * s + delta * f
    P = Sigmoid(logit(P))

This is a standard baseline for Knowledge Tracing, often comparable to BKT.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

from models.base import KnowledgeTracingModel
from data.schemas import Dataset, StudentInteraction

class LogisticModel(KnowledgeTracingModel):
    """
    Performance Factor Analysis (PFA) / Logistic Regression Model.
    """
    
    def __init__(self):
        super().__init__()
        # Parameters per skill: {beta, gamma, delta}
        self.params = {} 
        
    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Fit PFA parameters using Maximum Likelihood Estimation.
        """
        # Group data by skill
        skill_data = {}
        
        for seq in dataset.sequences:
            student_history = {} # skill_id -> {s: count, f: count}
            
            for interaction in seq.interactions:
                sid = interaction.skill_id
                if sid not in skill_data:
                    skill_data[sid] = {'X': [], 'y': []}
                
                # Get current counters
                if sid not in student_history:
                    student_history[sid] = {'s': 0, 'f': 0}
                
                s = student_history[sid]['s']
                f = student_history[sid]['f']
                
                # Append features (1 for bias/beta, s, f)
                skill_data[sid]['X'].append([1.0, s, f])
                skill_data[sid]['y'].append(interaction.correct)
                
                # Update counters
                if interaction.correct:
                    student_history[sid]['s'] += 1
                else:
                    student_history[sid]['f'] += 1
        
        print(f"Fitting Logistic Model (PFA) for {len(skill_data)} skills...")
        
        # Optimize for each skill
        for skill_id, data in skill_data.items():
            X = np.array(data['X'])
            y = np.array(data['y'])
            
            # Init params: beta=0, gamma=0.1, delta=0.1
            initial_guess = [0.0, 0.1, 0.1]
            
            res = minimize(
                self._negative_log_likelihood,
                initial_guess,
                args=(X, y),
                method='L-BFGS-B'
            )
            
            self.params[skill_id] = {
                'beta': res.x[0],
                'gamma': res.x[1],
                'delta': res.x[2]
            }
            
        self.is_fitted = True

    def _negative_log_likelihood(self, params, X, y):
        # params: [beta, gamma, delta]
        # X: [N, 3] matrix of [1, s, f]
        
        logits = np.dot(X, params)
        probs = expit(logits)
        
        # Clip probabilities
        probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
        
        # NLL
        nll = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        return nll

    def predict_next(self, student_id: str, history: List[StudentInteraction], next_skill: str) -> float:
        """Predict probability for next item."""
        if next_skill not in self.params:
            return 0.5
            
        params = self.params[next_skill]
        beta, gamma, delta = params['beta'], params['gamma'], params['delta']
        
        # Calculate s and f from history
        s, f = 0, 0
        for intr in history:
            if intr.skill_id == next_skill:
                if intr.correct:
                    s += 1
                else:
                    f += 1
        
        logit = beta + gamma * s + delta * f
        return expit(logit)

    def get_knowledge_state(self, student_id: str, skill_id: str, history: List[StudentInteraction]) -> float:
        # PFA doesn't have a latent "knowledge state" like BKT.
        # But predicted probability is often used as a proxy.
        return self.predict_next(student_id, history, skill_id)
