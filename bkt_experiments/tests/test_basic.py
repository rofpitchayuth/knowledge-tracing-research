"""
Simple test to verify BKT experiments framework is working.

This tests basic imports and functionality.
Run: python -m pytest tests/test_basic.py -v
Or:  python tests/test_basic.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    from data.mock_generator import MockDataGenerator, LearnerProfile
    from models.bkt.standard_bkt import StandardBKT, BKTParameters
    from data.schemas import StudentInteraction, StudentSequence, Dataset
    from analysis.metrics import compute_all_metrics
    
    print("✓ All imports successful")


def test_data_generation():
    """Test mock data generation."""
    from data.mock_generator import MockDataGenerator
    
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_dataset(
        num_students=10,
        num_skills=2,
        min_attempts_per_student=10,
        max_attempts_per_student=20
    )
    
    assert dataset.num_students == 10
    assert len(dataset.skills) == 2
    assert dataset.num_interactions > 0
    
    print(f"✓ Data generation works: {dataset.num_students} students, "
          f"{dataset.num_interactions} interactions")


def test_bkt_model():
    """Test BKT model fitting and evaluation."""
    from data.mock_generator import MockDataGenerator
    from models.bkt.standard_bkt import StandardBKT
    
    # Generate small dataset
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_dataset(
        num_students=20,
        num_skills=2,
        min_attempts_per_student=15,
        max_attempts_per_student=25
    )
    
    # Create and fit model
    model = StandardBKT()
    model.fit(dataset, max_iterations=10, verbose=False)
    
    assert model.is_fitted
    assert len(model.skills_params) > 0
    
    # Evaluate
    metrics = model.evaluate(dataset)
    assert 'auc' in metrics
    assert 0.0 <= metrics['auc'] <= 1.0
    
    print(f"✓ BKT model works: AUC = {metrics['auc']:.4f}")


def test_parameter_sensitivity():
    """Test parameter sensitivity experiment."""
    from data.mock_generator import MockDataGenerator
    from models.bkt.standard_bkt import StandardBKT, BKTParameters
    from experiments.parameter_sensitivity import ParameterSensitivityExperiment
    import numpy as np
    import tempfile
    import os
    
    # Generate small dataset
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_dataset(
        num_students=20,
        num_skills=2,
        min_attempts_per_student=10,
        max_attempts_per_student=20
    )
    
    # Create temp output dir
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = ParameterSensitivityExperiment(output_dir=tmpdir)
        
        # Run small parameter sweep
        df = exp.run_single_parameter_sweep(
            parameter_name='p_learn',
            param_range=np.linspace(0.1, 0.3, 3).tolist(),
            dataset=dataset,
            num_trials=1,
            verbose=False
        )
        
        assert len(df) == 3
        assert 'value' in df.columns
        assert 'auc' in df.columns
    
    print("✓ Parameter sensitivity analysis works")


def test_metrics():
    """Test evaluation metrics."""
    from analysis.metrics import compute_all_metrics
    import numpy as np
    
    # Create fake predictions and targets
    np.random.seed(42)
    predictions = np.random.random(100)
    targets = np.random.randint(0, 2, 100)
    
    metrics = compute_all_metrics(predictions, targets)
    
    assert 'auc' in metrics
    assert 'accuracy' in metrics
    assert 'rmse' in metrics
    
    print(f"✓ Metrics calculation works: AUC = {metrics['auc']:.4f}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running BKT Experiments Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("BKT Model", test_bkt_model),
        ("Parameter Sensitivity", test_parameter_sensitivity),
        ("Metrics", test_metrics)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nTest: {name}")
            print("-" * 60)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All tests passed! Framework is working correctly.")
    else:
        print(f"⚠️ {failed} test(s) failed.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
