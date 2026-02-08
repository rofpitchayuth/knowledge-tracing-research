"""
Deep Knowledge Tracing (DKT)

LSTM-based deep learning model for knowledge tracing.
Instead of explicitly modeling P(L), P(T), P(G), P(S) like BKT,
DKT learns a continuous latent representation of student knowledge.

Based on: Piech et al. (2015) - Deep Knowledge Tracing
https://arxiv.org/abs/1506.05908

Key Differences from BKT:
- Continuous latent state (LSTM hidden state) vs discrete learned/not-learned
- No explicit parameters (black-box model)
- Can capture complex temporal patterns
- Requires more data to train
- Less interpretable but potentially more accurate

Trade-offs:
+ Can model complex learning patterns
+ No parameter constraints needed
+ Scales to many skills automatically
- Black-box (no interpretability)
- Requires more data (hundreds of students)
- Needs GPU for efficient training
- Harder to debug/understand predictions
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm

# Direct imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.base import KnowledgeTracingModel
from data.schemas import StudentInteraction, StudentSequence, Dataset


class DKTDataset(TorchDataset):
    """PyTorch Dataset for DKT training."""
    
    def __init__(self, sequences: List[StudentSequence], num_skills: int):
        """
        Args:
            sequences: List of student sequences
            num_skills: Total number of skills
        """
        self.sequences = sequences
        self.num_skills = num_skills
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            inputs: (seq_len, num_skills * 2) - one-hot encoding of (skill, correctness)
            targets: (seq_len,) - correctness of next attempt
            skills: (seq_len,) - skill IDs for next attempt
            mask: (seq_len,) - valid positions (for variable length sequences)
        """
        sequence = self.sequences[idx]
        interactions = sequence.interactions
        
        seq_len = len(interactions) - 1  # We predict next, so len-1
        
        if seq_len == 0:
            # Handle sequences with only 1 interaction
            seq_len = 1
            interactions = interactions + interactions  # Duplicate
        
        # Input: one-hot encoding of (skill, correctness) pairs
        # Format: [skill_0_incorrect, skill_0_correct, skill_1_incorrect, ...]
        input_dim = self.num_skills * 2
        inputs = np.zeros((seq_len, input_dim), dtype=np.float32)
        targets = np.zeros(seq_len, dtype=np.float32)
        skills = np.zeros(seq_len, dtype=np.long)
        
        # Build input/target pairs
        for t in range(seq_len):
            current = interactions[t]
            next_interaction = interactions[t + 1]
            
            # Input: current (skill, correctness)
            skill_id = int(current.skill_id.split('_')[1])  # Extract number from 'skill_XX'
            idx_offset = skill_id * 2
            
            if current.correct == 1:
                inputs[t, idx_offset + 1] = 1  # Skill correct
            else:
                inputs[t, idx_offset] = 1  # Skill incorrect
            
            # Target: next correctness
            targets[t] = float(next_interaction.correct)
            skills[t] = int(next_interaction.skill_id.split('_')[1])
        
        return (
            torch.FloatTensor(inputs),
            torch.FloatTensor(targets),
            torch.LongTensor(skills),
            torch.ones(seq_len)  # Mask (all valid)
        )


class DKTModel(nn.Module):
    """LSTM-based Deep Knowledge Tracing model."""
    
    def __init__(self, num_skills: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2):
        """
        Args:
            num_skills: Number of skills/knowledge components
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(DKTModel, self).__init__()
        
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        input_size = num_skills * 2  # (skill, correctness) pairs
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict probability of correctness for each skill
        self.fc = nn.Linear(hidden_size, num_skills)
        
        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, seq_len, num_skills * 2)
            hidden: Optional initial hidden state
            
        Returns:
            predictions: (batch_size, seq_len, num_skills)
            hidden: Final hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear projection to skills
        logits = self.fc(lstm_out)
        
        # Sigmoid for probabilities
        predictions = self.sigmoid(logits)
        
        return predictions, hidden


class DeepKnowledgeTracing(KnowledgeTracingModel):
    """
    Deep Knowledge Tracing model wrapper.
    
    Implements the KnowledgeTracingModel interface for consistency
    with BKT models.
    """
    
    def __init__(self, num_skills: Optional[int] = None,
                 hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2, device: str = 'cpu'):
        """
        Initialize DKT model.
        
        Args:
            num_skills: Number of skills (will be inferred from data if None)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.model_name = "Deep Knowledge Tracing (DKT)"
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.model = None
        self.skill_map = {}  # Map skill_id string to integer index
        self.reverse_skill_map = {}
    
    def _build_skill_map(self, dataset: Dataset):
        """Create mapping between skill IDs and integer indices."""
        skill_ids = sorted(dataset.get_skill_ids())
        self.skill_map = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
        self.reverse_skill_map = {idx: skill_id for skill_id, idx in self.skill_map.items()}
        self.num_skills = len(skill_ids)
    
    def fit(self, dataset: Dataset, epochs: int = 20, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = False, **kwargs) -> None:
        """
        Train DKT model.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print training progress
        """
        if verbose:
            print(f"Training {self.model_name}...")
            print(f"  Students: {dataset.num_students}")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
        
        # Build skill mapping
        self._build_skill_map(dataset)
        
        # Create PyTorch dataset
        torch_dataset = DKTDataset(dataset.sequences, self.num_skills)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = DKTModel(
            num_skills=self.num_skills,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training DKT")
        
        for epoch in iterator:
            total_loss = 0.0
            num_batches = 0
            
            for inputs, targets, skills, masks in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                skills = skills.to(self.device)
                
                # Forward pass
                predictions, _ = self.model(inputs)
                
                # Extract predictions for the specific skills
                batch_size, seq_len = skills.shape
                batch_predictions = torch.zeros(batch_size, seq_len).to(self.device)
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        skill_idx = skills[b, t].item()
                        batch_predictions[b, t] = predictions[b, t, skill_idx]
                
                # Compute loss
                loss = criterion(batch_predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
        if verbose:
            print(f"Training completed. Final loss: {avg_loss:.4f}")
    
    def predict_next(self, student_id: str, history: List[StudentInteraction],
                     next_skill: str) -> float:
        """
        Predict probability of correct answer for next interaction.
        
        Args:
            student_id: Student ID (unused in DKT)
            history: Past interactions
            next_skill: Skill for next question
            
        Returns:
            Probability of correct answer
        """
        if not self.is_fitted or self.model is None:
            return 0.5  # Random guess if not fitted
        
        if len(history) == 0:
            return 0.5  # No history → random guess
        
        self.model.eval()
        
        with torch.no_grad():
            # Build input sequence
            seq_len = len(history)
            input_dim = self.num_skills * 2
            inputs = np.zeros((1, seq_len, input_dim), dtype=np.float32)
            
            for t, interaction in enumerate(history):
                if interaction.skill_id not in self.skill_map:
                    continue
                
                skill_idx = self.skill_map[interaction.skill_id]
                idx_offset = skill_idx * 2
                
                if interaction.correct == 1:
                    inputs[0, t, idx_offset + 1] = 1
                else:
                    inputs[0, t, idx_offset] = 1
            
            # Forward pass
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)
            predictions, _ = self.model(inputs_tensor)
            
            # Get prediction for next skill (last timestep)
            if next_skill in self.skill_map:
                next_skill_idx = self.skill_map[next_skill]
                p_correct = predictions[0, -1, next_skill_idx].item()
            else:
                p_correct = 0.5
            
            return float(np.clip(p_correct, 0.0, 1.0))
    
    def get_knowledge_state(self, student_id: str, skill_id: str,
                           history: List[StudentInteraction]) -> float:
        """
        Get knowledge state (interpreted as P(correct) for that skill).
        
        Note: DKT doesn't have explicit knowledge states like BKT.
        We approximate it as the current prediction probability.
        """
        return self.predict_next(student_id, history, skill_id)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_skills': self.num_skills,
                'skill_map': self.skill_map,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.num_skills = checkpoint['num_skills']
        self.skill_map = checkpoint['skill_map']
        self.reverse_skill_map = {v: k for k, v in self.skill_map.items()}
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        
        self.model = DKTModel(
            num_skills=self.num_skills,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
