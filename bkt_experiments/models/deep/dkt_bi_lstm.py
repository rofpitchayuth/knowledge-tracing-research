"""
Bidirectional LSTM Deep Knowledge Tracing.
"""

import torch
import torch.nn as nn
from typing import Optional

# Direct imports from dkt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.deep.dkt import DeepKnowledgeTracing, DKTModel

class DKTBiLSTMModel(DKTModel):
    """Bidirectional LSTM model for knowledge tracing."""
    
    def __init__(self, num_skills: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2):
        super(DKTModel, self).__init__()  # Initialize nn.Module directly, skip DKTModel init to avoid double init
        
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        input_size = num_skills * 2
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: hidden_size * 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_skills)
        self.sigmoid = nn.Sigmoid()

class DeepKnowledgeTracingBiLSTM(DeepKnowledgeTracing):
    """
    Bi-LSTM based Knowledge Tracing.
    """
    
    def __init__(self, num_skills: Optional[int] = None,
                 hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2, device: str = 'cpu'):
        super().__init__(num_skills, hidden_size, num_layers, dropout, device)
        self.model_name = "DKT (Bi-LSTM)"
        
    def _init_model(self, num_skills: int) -> nn.Module:
        return DKTBiLSTMModel(
            num_skills=num_skills,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
