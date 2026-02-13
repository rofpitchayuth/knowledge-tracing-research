import argparse
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.model_comparison import ModelComparison
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting
from models.bkt.individualized_bkt import IndividualizedBKT
from models.bkt.improved_bkt import ImprovedBKT
from models.logistic.logistic_model import LogisticModel

from models.deep.dkt import DeepKnowledgeTracing

from data.mock_generator import MockDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Compare BKT variants, Logistic Regression and Deep Learning'
    )
    parser.add_argument(
        '--students',
        type=int,
        default=300,
        help='Number of students (default: 300, recommend 300+ for DKT)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='DKT training epochs (default: 15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/complete_TIMESTAMP)'
    )
    parser.add_argument(
        '--skip-dkt',
        action='store_true',
        help='Skip DKT (faster, for testing)'
    )
    
    args = parser.parse_args()
    
    # Generate output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/complete_{timestamp}"
    else:
        output_dir = args.output
    
    print(f"Configuration:")
    print(f"  Students: {args.students}") 
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print()
    
    # Generate dataset
    print("Generating synthetic dataset...")
    generator = MockDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(
        num_students=args.students,
        num_skills=10,
        min_attempts_per_student=30,
        max_attempts_per_student=50,
        forgetting_rate=0.05
    )
    
    print(f"Dataset: {dataset.num_students} students, {dataset.num_interactions} interactions")
    print()
    
    # Create comparison framework
    comparison = ModelComparison(output_dir=output_dir)
    
    # Add BKT models
    print("Adding models...")
    comparison.add_model("Standard BKT", StandardBKT())
    comparison.add_model("BKT with Forgetting", BKTWithForgetting())
    comparison.add_model("Individualized BKT", IndividualizedBKT())
    comparison.add_model("Improved BKT (Time)", ImprovedBKT())
    comparison.add_model("Logistic Model (PFA)", LogisticModel())
    
    # Add Deep Learning
    if not args.skip_dkt:
        dkt = DeepKnowledgeTracing(hidden_size=128, num_layers=1, dropout=0.2)
        comparison.add_model("Deep Knowledge Tracing", dkt)
        print("  (DKT will train for {} epochs - this may take a few minutes)".format(args.epochs))
    
    print()
    
    # Run comparison
    print(" RUNNING COMPARISON")
    
    # Custom fit params for DKT
    fit_params = {
        'max_iterations': 50,
        'verbose': False,
        'epochs': args.epochs,  # For DKT
        'batch_size': 32,  # For DKT
        'learning_rate': 0.001  # For DKT
    }
    
    results_df = comparison.compare_on_dataset(
        dataset,
        fit_params=fit_params,
        verbose=True
    )
    
    # Data efficiency (skip DKT for speed)
    print(" DATA EFFICIENCY ANALYSIS")
    
    efficiency_df = comparison.compare_data_efficiency(
        dataset,
        sample_sizes=[50, 100, 200, 300],
        num_trials=2,
        verbose=True
    )
    
    # Save summary
    comparison.save_summary_report()
    
    print(f"\nResults saved to: {output_dir}/")
    print()
    
    # Detailed summary
    print(" KEY FINDINGS")
    
    # Best models
    best_by_auc = results_df.loc[results_df['test_auc'].idxmax()]
    best_by_speed = results_df.loc[results_df['training_time_seconds'].idxmin()]
    
    print(f"\n Best Accuracy: {best_by_auc['model']}")
    print(f"   AUC: {best_by_auc['test_auc']:.4f}")
    print(f"   Accuracy: {best_by_auc['test_accuracy']:.4f}")
    print(f"   Training time: {best_by_auc['training_time_seconds']:.2f}s")
    
    print(f"\n Fastest: {best_by_speed['model']}")
    print(f"   Training time: {best_by_speed['training_time_seconds']:.2f}s")
    print(f"   AUC: {best_by_speed['test_auc']:.4f}")
    
if __name__ == "__main__":
    main()
