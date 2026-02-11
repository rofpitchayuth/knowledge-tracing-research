"""
Run Complete Model Comparison: BKT vs Deep Learning

Compare all implemented models:
- Standard BKT (4 parameters)
- BKT with Forgetting (5 parameters)
- Individualized BKT (per-student parameters)
- Deep Knowledge Tracing (LSTM-based neural network)

This answers the ultimate research question:
"Traditional Knowledge Tracing vs Modern Deep Learning - which is better?"

Usage:
    python run_complete_comparison.py
    
Or with custom settings:
    python run_complete_comparison.py --students 300 --epochs 15 --output results/complete
"""

import argparse
from datetime import datetime
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.model_comparison import ModelComparison
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting
from models.bkt.individualized_bkt import IndividualizedBKT
from models.bkt.improved_bkt import ImprovedBKT
from models.logistic.logistic_model import LogisticModel

# Try to import Deep Learning models
from models.deep.dkt import DeepKnowledgeTracing

from data.mock_generator import MockDataGenerator


def main():
    """Main function to run complete comparison."""
    parser = argparse.ArgumentParser(
        description='Compare BKT variants and Deep Learning'
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
    
    print("\n" + "="*70)
    print(" COMPLETE MODEL COMPARISON: BKT vs DEEP LEARNING")
    print("="*70)
    print(f"\nModels to compare:")
    print(f"  📊 Traditional BKT:")
    print(f"     1. Standard BKT (4 parameters)")
    print(f"     2. BKT with Forgetting (5 parameters)")
    print(f"     3. Individualized BKT (per-student)")
    if not args.skip_dkt:
        print(f"  🤖 Deep Learning:")
        print(f"     4. Deep Knowledge Tracing (LSTM, {args.epochs} epochs)")
    print()
    print(f"Configuration:")
    print(f"  Students: {args.students} {'(⚠️ DKT works best with 500+)' if args.students < 500 and not args.skip_dkt else ''}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print()
    
    # Generate dataset
    print("Generating synthetic dataset...")
    generator = MockDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(
        num_students=args.students,
        num_skills=3,
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
    print("="*70)
    print(" RUNNING COMPARISON")
    print("="*70)
    
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
    print("\n" + "="*70)
    print(" DATA EFFICIENCY ANALYSIS")
    print("="*70)
    print("(Testing BKT models only for speed)")
    
    # Temporarily remove DKT for efficiency test
    dkt_model = comparison.models.pop("Deep Knowledge Tracing", None)
    
    efficiency_df = comparison.compare_data_efficiency(
        dataset,
        sample_sizes=[50, 100, 200],
        num_trials=2,
        verbose=True
    )
    
    # Add DKT back
    if dkt_model:
        comparison.models["Deep Knowledge Tracing"] = dkt_model
    
    # Save summary
    comparison.save_summary_report()
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print()
    
    # Detailed summary
    print("="*70)
    print(" KEY FINDINGS")
    print("="*70)
    
    # Best models
    best_by_auc = results_df.loc[results_df['test_auc'].idxmax()]
    best_by_speed = results_df.loc[results_df['training_time_seconds'].idxmin()]
    
    print(f"\n🏆 Best Accuracy: {best_by_auc['model']}")
    print(f"   AUC: {best_by_auc['test_auc']:.4f}")
    print(f"   Accuracy: {best_by_auc['test_accuracy']:.4f}")
    print(f"   Training time: {best_by_auc['training_time_seconds']:.2f}s")
    
    print(f"\n⚡ Fastest: {best_by_speed['model']}")
    print(f"   Training time: {best_by_speed['training_time_seconds']:.2f}s")
    print(f"   AUC: {best_by_speed['test_auc']:.4f}")
    
    # Interpretability note
    print(f"\n💡 Interpretability:")
    print(f"   ✅ BKT models: Highly interpretable (explicit parameters)")
    print(f"   ❌ DKT: Black-box (no parameter interpretation)")
    
    # Recommendations
    print("\n" + "="*70)
    print(" RECOMMENDATIONS")
    print("="*70)
    
    bkt_best = results_df[results_df['model'].str.contains('BKT')]['test_auc'].max()
    dkt_auc = results_df[results_df['model'] == 'Deep Knowledge Tracing']['test_auc'].values
    dkt_auc = dkt_auc[0] if len(dkt_auc) > 0 else 0
    
    if dkt_auc > bkt_best + 0.02:
        print("\n✅ Deep Learning (DKT) wins!")
        print(f"   AUC improvement: {(dkt_auc - bkt_best):.4f} ({((dkt_auc - bkt_best)/bkt_best*100):.1f}%)")
        print("   → Use DKT if accuracy is priority and interpretability not needed")
    elif bkt_best > dkt_auc + 0.01:
        print("\n✅ BKT wins!")
        print(f"   BKT is both more accurate AND interpretable")
        print("   → Use BKT (Forgetting or Individualized)")
    else:
        print("\n⚖️  Close competition!")
        print(f"   Difference: {abs(dkt_auc - bkt_best):.4f} (< 2%)")
        print("   → Choose based on needs:")
        print("      - Need interpretability? → BKT")
        print("      - Need max accuracy? → DKT")
        print("      - Need speed? → Standard BKT")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print(f"\n1. View results:")
    print(f"   - {output_dir}/comparison_results.csv")
    print(f"   - {output_dir}/comparison_summary.json")
    print()
    print(f"2. Create visualizations:")
    print(f"   python create_comparison_plots.py --input {output_dir}")
    print()
    print("3. Write your research paper with these findings!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
