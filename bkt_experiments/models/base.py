"""
Base classes for Knowledge Tracing models.

This module defines abstract base classes that all KT models should implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np

# Direct import instead of relative
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schemas import StudentInteraction, StudentSequence, Dataset


class KnowledgeTracingModel(ABC):
    """Abstract base class for all Knowledge Tracing models."""
    
    def __init__(self, **kwargs):
        """Initialize the model with optional hyperparameters."""
        self.model_name = self.__class__.__name__
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Train/fit the model on a dataset.
        
        Args:
            dataset: Training dataset containing student sequences
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict_next(self, student_id: str, history: List[StudentInteraction], 
                     next_skill: str) -> float:
        """
        Predict the probability of answering the next question correctly.
        
        Args:
            student_id: Student identifier
            history: List of previous interactions for this student
            next_skill: Skill ID of the next question
            
        Returns:
            Probability of answering correctly (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_knowledge_state(self, student_id: str, skill_id: str, 
                           history: List[StudentInteraction]) -> float:
        """
        Get the current knowledge state/mastery level for a skill.
        
        Args:
            student_id: Student identifier
            skill_id: Skill identifier
            history: List of previous interactions
            
        Returns:
            Knowledge state (interpretation depends on model)
        """
        pass
    
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = []
        targets = []
        
        for sequence in dataset.sequences:
            for i in range(1, len(sequence.interactions)):
                history = sequence.interactions[:i]
                current = sequence.interactions[i]
                
                pred = self.predict_next(
                    sequence.student_id,
                    history,
                    current.skill_id
                )
                
                predictions.append(pred)
                targets.append(current.correct)
        
        return self._compute_metrics(np.array(predictions), np.array(targets))
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
        
        # Binary predictions using 0.5 threshold
        binary_preds = (predictions >= 0.5).astype(int)
        
        metrics = {
            'auc': float(roc_auc_score(targets, predictions)),
            'accuracy': float(accuracy_score(targets, binary_preds)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'log_loss': float(log_loss(targets, predictions, labels=[0, 1])),
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """Save model parameters to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filepath: str) -> None:
        """Load model parameters from file."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)


class ParametricBKTModel(KnowledgeTracingModel):
    """Base class for parametric BKT variants with explicit parameters."""
    
    @abstractmethod
    def get_parameters(self, skill_id: str) -> Dict[str, float]:
        """
        Get model parameters for a specific skill.
        
        Args:
            skill_id: Skill identifier
            
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, skill_id: str, params: Dict[str, float]) -> None:
        """
        Set model parameters for a specific skill.
        
        Args:
            skill_id: Skill identifier
            params: Dictionary of parameter names and values
        """
        pass
