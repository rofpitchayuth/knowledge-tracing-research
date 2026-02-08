"""
Quick Demo Script for BKT Experiments

This script demonstrates the basic functionality of the BKT experimental framework.
Run this to verify installation and see example output.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path if needed
# (This ensures we can import the modules even when running from different directories)
sys.path.insert(0, str(Path(__file__).parent))

# Import using absolute imports from the bkt_experiments package
try:
    # Try package imports first (when installed or in correct path)
    from bkt_experiments.data.mock_generator import MockDataGenerator, LearnerProfile
    from bkt_experiments.models.bkt.standard_bkt import StandardBKT, BKTParameters
    from bkt_experiments.analysis.metrics import compute_all_metrics, print_metric_summary
except ImportError:
    # Fallback to direct imports (when running from bkt_experiments directory)
    from data.mock_generator import MockDataGenerator, LearnerProfile
    from models.bkt.standard_bkt import StandardBKT, BKTParameters
    from analysis.metrics import compute_all_metrics, print_metric_summary


def demo_data_generation():
    """Demonstrate mock data generation."""
    print("\n" + "="*70)
    print("DEMO 1: Mock Data Generation")
    print("="*70)
    
    generator = MockDataGenerator(seed=42)
    
    # Generate a small dataset
    dataset = generator.generate_dataset(
        num_students=50,
        num_skills=3,
        skill_names=["Limits", "Derivatives", "Integrals"],
        min_attempts_per_student=20,
        max_attempts_per_student=40,
        include_timestamps=True
    )
    
    print(f"\nGenerated dataset:")
    print(f"  Students: {dataset.num_students}")
    print(f"  Skills: {len(dataset.skills)}")
    print(f"  Total interactions: {dataset.num_interactions}")
    print(f"  Items: {len(dataset.items)}")
    
    print(f"\nSkills:")
    for skill_id, skill in dataset.skills.items():
        print(f"  - {skill.name} ({skill_id})")
    
    print(f"\nLearner profiles in dataset:")
    profiles = {}
    for seq in dataset.sequences:
        profile = seq.metadata.get('profile', 'Unknown')
        profiles[profile] = profiles.get(profile, 0) + 1
    
    for profile, count in profiles.items():
        print(f"  {profile}: {count} students")
    
    print(f"\nSample interaction from first student:")
    first_student = dataset.sequences[0]
    interaction = first_student.interactions[0]
    print(f"  Student: {interaction.student_id}")
    print(f"  Item: {interaction.item_id}")
    print(f"  Skill: {interaction.skill_id}")
    print(f"  Correct: {interaction.correct}")
    print(f"  Time taken: {interaction.time_taken_seconds:.1f}s")
    
    return dataset


def demo_bkt_model(dataset):
    """Demonstrate BKT model fitting and evaluation."""
    print("\n" + "="*70)
    print("DEMO 2: Standard BKT Model")
    print("="*70)
    
    # Create model
    model = StandardBKT()
    
    print("\nFitting BKT model using EM algorithm...")
    model.fit(dataset, max_iterations=50, tolerance=1e-4, verbose=True)
    
    # Show learned parameters
    print("\nLearned parameters:")
    for skill_id in sorted(model.skills_params.keys()):
        params = model.get_parameters(skill_id)
        skill_name = dataset.skills[skill_id].name
        print(f"\n  {skill_name} ({skill_id}):")
        print(f"    P(L0) = {params['p_init']:.4f}  (initial knowledge)")
        print(f"    P(T)  = {params['p_learn']:.4f}  (learning rate)")
        print(f"    P(G)  = {params['p_guess']:.4f}  (guess rate)")
        print(f"    P(S)  = {params['p_slip']:.4f}  (slip rate)")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(dataset)
    print_metric_summary(metrics, title="BKT Model Performance")
    
    return model


def demo_predictions(model, dataset):
    """Demonstrate making predictions."""
    print("\n" + "="*70)
    print("DEMO 3: Making Predictions")
    print("="*70)
    
    # Get first student
    student = dataset.sequences[0]
    print(f"\nStudent: {student.student_id}")
    print(f"Profile: {student.metadata.get('profile')}")
    print(f"Total interactions: {len(student.interactions)}")
    
    # Use first 10 interactions as history
    history = student.interactions[:10]
    next_interaction = student.interactions[10]
    
    print(f"\nPredicting for interaction {len(history)+1}:")
    print(f"  Skill: {next_interaction.skill_id}")
    print(f"  Actual result: {'Correct' if next_interaction.correct else 'Incorrect'}")
    
    # Make prediction
    p_correct = model.predict_next(
        student.student_id,
        history,
        next_interaction.skill_id
    )
    
    print(f"  Predicted P(Correct): {p_correct:.4f}")
    
    # Get knowledge state
    p_mastery = model.get_knowledge_state(
        student.student_id,
        next_interaction.skill_id,
        history
    )
    
    print(f"  P(Mastered): {p_mastery:.4f}")
    
    # Show learning progression
    print(f"\nLearning progression for {next_interaction.skill_id}:")
    for i in range(min(15, len(student.interactions))):
        history_i = student.interactions[:i] if i > 0 else []
        p_know = model.get_knowledge_state(
            student.student_id,
            next_interaction.skill_id,
            history_i
        )
        interaction = student.interactions[i]
        if interaction.skill_id == next_interaction.skill_id:
            result_str = "✓" if interaction.correct else "✗"
            print(f"  Attempt {i+1}: P(Know) = {p_know:.3f}, Result = {result_str}")


def demo_custom_parameters(dataset):
    """Demonstrate setting custom parameters."""
    print("\n" + "="*70)
    print("DEMO 4: Custom Parameters")
    print("="*70)
    
    # Create model with custom parameters
    model = StandardBKT()
    
    # Set expert-defined parameters
    expert_params = {
        'p_init': 0.15,   # Low initial knowledge
        'p_learn': 0.25,  # High learning rate
        'p_guess': 0.18,  # Moderate guessing
        'p_slip': 0.08    # Low slip
    }
    
    print("\nSetting custom (expert-defined) parameters:")
    print(f"  {expert_params}")
    
    for skill_id in dataset.get_skill_ids():
        model.set_parameters(skill_id, expert_params)
    
    model.is_fitted = True
    
    # Evaluate
    metrics = model.evaluate(dataset)
    print_metric_summary(metrics, title="Custom Parameters Performance")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" BKT EXPERIMENTS - QUICK DEMO")
    print("="*70)
    print("\nThis script demonstrates the basic functionality of the framework.")
    
    # Run demos
    dataset = demo_data_generation()
    model = demo_bkt_model(dataset)
    demo_predictions(model, dataset)
    demo_custom_parameters(dataset)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Explore notebooks/01_quick_start.ipynb for interactive examples")
    print("  2. Run parameter sensitivity analysis (see README.md)")
    print("  3. Create visualizations with analysis/visualizations.py")
    print("  4. Read README.md for full documentation")
    print()


if __name__ == "__main__":
    main()
