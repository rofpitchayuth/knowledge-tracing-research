"""
Test script for verifying Improved BKT and Logistic Model implementations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mock_generator import MockDataGenerator
from models.bkt.improved_bkt import ImprovedBKT
from models.logistic.logistic_model import LogisticModel

class TestNewModels(unittest.TestCase):
    
    def setUp(self):
        # Generate small dataset
        self.generator = MockDataGenerator(seed=42)
        self.dataset = self.generator.generate_dataset(
            num_students=20,
            num_skills=2,
            min_attempts_per_student=10,
            max_attempts_per_student=20,
            include_timestamps=True
        )
        
    def test_improved_bkt(self):
        print("\nTesting Improved BKT...")
        model = ImprovedBKT()
        
        # Test fitting
        model.fit(self.dataset)
        self.assertTrue(model.is_fitted)
        self.assertTrue(len(model.params) > 0)
        
        # Check params structure
        first_skill = list(model.params.keys())[0]
        params = model.params[first_skill]
        required = ['p_init', 'p_learn', 'w_s', 'b_s', 'w_g', 'b_g']
        for p in required:
            self.assertIn(p, params)
            
        print(f"Improved BKT Params for {first_skill}: {params}")

        # Test prediction
        seq = self.dataset.sequences[0]
        history = seq.interactions[:5]
        next_skill = seq.interactions[5].skill_id
        
        pred = model.predict_next(seq.student_id, history, next_skill)
        self.assertTrue(0.0 <= pred <= 1.0)
        print(f"Prediction: {pred}")

    def test_logistic_model(self):
        print("\nTesting Logistic Model (PFA)...")
        model = LogisticModel()
        
        # Test fitting
        model.fit(self.dataset)
        self.assertTrue(model.is_fitted)
        self.assertTrue(len(model.params) > 0)
        
        # Check params structure
        first_skill = list(model.params.keys())[0]
        params = model.params[first_skill]
        required = ['beta', 'gamma', 'delta']
        for p in required:
            self.assertIn(p, params)
            
        print(f"Logistic Model Params for {first_skill}: {params}")
        
        # Test prediction
        seq = self.dataset.sequences[0]
        history = seq.interactions[:5]
        next_skill = seq.interactions[5].skill_id
        
        pred = model.predict_next(seq.student_id, history, next_skill)
        self.assertTrue(0.0 <= pred <= 1.0)
        print(f"Prediction: {pred}")

if __name__ == '__main__':
    unittest.main()
