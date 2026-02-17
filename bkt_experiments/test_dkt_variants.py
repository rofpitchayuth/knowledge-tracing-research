
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.mock_generator import MockDataGenerator
from models.deep.dkt import DeepKnowledgeTracing
from models.deep.dkt_bi_lstm import DeepKnowledgeTracingBiLSTM
from models.deep.dkt_gru import DeepKnowledgeTracingGRU
import torch
import traceback

def test_dkt_variants():
    print("Generating data...")
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_dataset(
        num_students=50,
        num_skills=5,
        min_attempts_per_student=10,
        max_attempts_per_student=50
    )
    
    print(f"Data generated: {dataset.num_students} students")
    
    models = [
        ("Standard DKT (LSTM)", DeepKnowledgeTracing(hidden_size=32, device='cpu')),
        ("Bi-LSTM DKT", DeepKnowledgeTracingBiLSTM(hidden_size=32, device='cpu')),
        ("GRU DKT", DeepKnowledgeTracingGRU(hidden_size=32, device='cpu'))
    ]
    
    for name, model in models:
        print(f"\nTesting {name}...")
        try:
            model.fit(dataset, epochs=2, batch_size=5, verbose=True)
            print(f"{name} Fit Success!")
            
            # Simple prediction test
            p = model.predict_next("student_1", dataset.sequences[0].interactions[:5], "skill_0")
            print(f"Prediction test: {p:.4f}")
            
        except Exception as e:
            print(f"{name} Failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_dkt_variants()
