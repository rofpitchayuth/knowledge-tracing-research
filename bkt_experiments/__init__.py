"""
BKT Experiments Package

A standalone research framework for analyzing Bayesian Knowledge Tracing models.

Quick Start:
    # When running from the bkt_experiments directory:
    from data.mock_generator import MockDataGenerator
    from models.bkt.standard_bkt import StandardBKT
    
    # Generate data
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_dataset(num_students=100)
    
    # Fit model
    model = StandardBKT()
    model.fit(dataset, verbose=True)
    
    # Evaluate
    metrics = model.evaluate(dataset)
    print(f"AUC: {metrics['auc']:.4f}")

For more examples, see:
    - demo.py
    - notebooks/01_quick_start.ipynb
    - README.md
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Convenience imports - only available when imported as a package
# (not when running scripts directly from this directory)
try:
    from .models.bkt.standard_bkt import StandardBKT, BKTParameters
    from .data.mock_generator import MockDataGenerator, LearnerProfile
    from .data.schemas import StudentInteraction, StudentSequence, Dataset, Skill, Item
    
    __all__ = [
        'StandardBKT',
        'BKTParameters',
        'MockDataGenerator',
        'LearnerProfile',
        'StudentInteraction',
        'StudentSequence',
        'Dataset',
        'Skill',
        'Item',
    ]
except ImportError:
    # Running as script from this directory - imports not available
    __all__ = []
