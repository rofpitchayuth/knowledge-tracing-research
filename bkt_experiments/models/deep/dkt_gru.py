"""
Gated Recurrent Unit (GRU) Deep Knowledge Tracing.
"""

import torch
import torch.nn as nn
from typing import Optional

# Direct imports from dkt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.deep.dkt import DeepKnowledgeTracing, DKTModel

class DKTGRUModel(nn.Module):
    """GRU model for knowledge tracing."""
    
    def __init__(self, num_skills: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2):
        super(DKTGRUModel, self).__init__()
        
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        input_size = num_skills * 2
        
        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_skills)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Dropout
        gru_out = self.dropout(gru_out)
        
        # Linear projection
        logits = self.fc(gru_out)
        
        # Sigmoid
        predictions = self.sigmoid(logits)
        
        return predictions, hidden

class DeepKnowledgeTracingGRU(DeepKnowledgeTracing):
    """
    GRU based Knowledge Tracing.
    """
    
    def __init__(self, num_skills: Optional[int] = None,
                 hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2, device: str = 'cpu'):
        super().__init__(num_skills, hidden_size, num_layers, dropout, device)
        self.model_name = "DKT (GRU)"
        
    def _init_model(self, num_skills: int) -> nn.Module:
        return DKTGRUModel(
            num_skills=num_skills,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
