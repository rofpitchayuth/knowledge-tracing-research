"""
Model Comparison Framework

This module provides tools to compare multiple Knowledge Tracing models:
- Standard BKT
- BKT with Forgetting
- Future: Individualized BKT, Contextual BKT, DKT, DKVMN, SAKT

Comparison dimensions:
1. Predictive accuracy (AUC, Accuracy, RMSE)
2. Calibration quality (ECE)
3. Computational efficiency (training time, memory)
4. Interpretability (parameter transparency)
5. Data efficiency (performance with limited data)
"""

from typing import Dict, List, Tuple, Optional, Type
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import time
from tqdm import tqdm

# Direct imports
import sys
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))
from models.base import KnowledgeTracingModel
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting
from data.schemas import Dataset
from data.mock_generator import MockDataGenerator
from analysis.metrics import compute_all_metrics


class ModelComparison:
    """
    Framework for comparing multiple Knowledge Tracing models.
    
    Provides systematic evaluation across multiple metrics and dimensions,
    with statistical significance testing and visualization support.
    """
    
    def __init__(self, output_dir: str = "results/model_comparison"):
        """
        Initialize model comparison framework.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.models = {}
        
    def add_model(self, name: str, model: KnowledgeTracingModel) -> None:
        """
        Add a model to the comparison.
        
        Args:
            name: Human-readable name for the model
            model: Instance of a KT model
        """
        self.models[name] = model
        print(f"Added model: {name}")
    
    def compare_on_dataset(
        self,
        dataset: Dataset,
        train_test_split: float = 0.8,
        fit_params: Optional[Dict] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare all models on a single dataset.
        
        Args:
            dataset: Dataset for evaluation
            train_test_split: Fraction of data for training (rest for testing)
            fit_params: Parameters for model fitting (max_iterations, etc.)
            verbose: Print progress
            
        Returns:
            DataFrame with comparison results
        """
        if verbose:
            print("\n" + "="*70)
            print(" MODEL COMPARISON")
            print("="*70)
            print(f"\nDataset: {dataset.num_students} students, "
                  f"{dataset.num_interactions} interactions")
            print(f"Train/Test split: {train_test_split:.0%}/{(1-train_test_split):.0%}")
            print()
        
        # Split dataset
        train_data, test_data = self._split_dataset(dataset, train_test_split)
        
        if verbose:
            print(f"Train: {train_data.num_students} students, "
                  f"{train_data.num_interactions} interactions")
            print(f"Test: {test_data.num_students} students, "
                  f"{test_data.num_interactions} interactions")
            print()
        
        if fit_params is None:
            fit_params = {'max_iterations': 50, 'verbose': False}
        
        # Evaluate each model
        results = []
        
        for model_name, model in self.models.items():
            if verbose:
                print(f"Evaluating: {model_name}")
                print("-" * 70)
            
            result = self._evaluate_single_model(
                model_name, model, train_data, test_data,
                fit_params, verbose
            )
            results.append(result)
            
            if verbose:
                print()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "comparison_results.csv"
        df.to_csv(output_file, index=False)
        
        if verbose:
            print("\n" + "="*70)
            print(" COMPARISON SUMMARY")
            print("="*70)
            self._print_comparison_summary(df)
            print(f"\nResults saved to: {output_file}")
        
        self.results['single_dataset'] = df
        return df
    
    def _evaluate_single_model(
        self,
        model_name: str,
        model: KnowledgeTracingModel,
        train_data: Dataset,
        test_data: Dataset,
        fit_params: Dict,
        verbose: bool
    ) -> Dict:
        """Evaluate a single model on train/test data."""
        result = {
            'model': model_name,
            'train_students': train_data.num_students,
            'test_students': test_data.num_students
        }
        
        try:
            # Training phase
            if verbose:
                print("  Training...")
            
            start_time = time.time()
            model.fit(train_data, **fit_params)
            training_time = time.time() - start_time
            
            result['training_time_seconds'] = training_time
            result['training_success'] = True
            
            if verbose:
                print(f"  ✓ Training completed in {training_time:.2f}s")
            
            # Evaluation on test set
            if verbose:
                print("  Evaluating on test set...")
            
            start_time = time.time()
            test_metrics = model.evaluate(test_data)
            inference_time = time.time() - start_time
            
            result['inference_time_seconds'] = inference_time
            result['test_accuracy'] = test_metrics['accuracy']
            result['test_auc'] = test_metrics['auc']
            result['test_rmse'] = test_metrics['rmse']
            result['test_log_loss'] = test_metrics['log_loss']
            result['test_ece'] = test_metrics.get('ece', np.nan)
            result['test_f1'] = test_metrics.get('f1', np.nan)
            
            # Train metrics (for overfitting check)
            train_metrics = model.evaluate(train_data)
            result['train_accuracy'] = train_metrics['accuracy']
            result['train_auc'] = train_metrics['auc']
            result['train_rmse'] = train_metrics['rmse']
            
            # Overfitting detection
            result['accuracy_gap'] = train_metrics['accuracy'] - test_metrics['accuracy']
            result['auc_gap'] = train_metrics['auc'] - test_metrics['auc']
            
            if verbose:
                print(f"  ✓ Test AUC: {test_metrics['auc']:.4f}")
                print(f"  ✓ Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  ✓ Test RMSE: {test_metrics['rmse']:.4f}")
            
        except Exception as e:
            result['training_success'] = False
            result['error'] = str(e)
            
            if verbose:
                print(f"  ✗ Error: {e}")
        
        return result
    
    def _split_dataset(self, dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and test sets at student level.
        
        Args:
            dataset: Full dataset
            train_ratio: Fraction of students for training
            
        Returns:
            (train_dataset, test_dataset)
        """
        num_train = int(len(dataset.sequences) * train_ratio)
        
        # Shuffle sequences for random split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(dataset.sequences))
        
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        # Create train dataset
        train_dataset = Dataset(
            sequences=[dataset.sequences[i] for i in train_indices],
            skills=dataset.skills,
            items=dataset.items
        )
        
        # Create test dataset
        test_dataset = Dataset(
            sequences=[dataset.sequences[i] for i in test_indices],
            skills=dataset.skills,
            items=dataset.items
        )
        
        return train_dataset, test_dataset
    
    def _print_comparison_summary(self, df: pd.DataFrame) -> None:
        """Print comparison summary table."""
        print("\nTest Set Performance:")
        print("-" * 70)
        
        # Create summary table
        summary_cols = ['model', 'test_auc', 'test_accuracy', 'test_rmse',
                       'training_time_seconds']
        
        if all(col in df.columns for col in summary_cols):
            summary = df[summary_cols].copy()
            summary = summary.sort_values('test_auc', ascending=False)
            
            # Format for display
            summary['test_auc'] = summary['test_auc'].apply(lambda x: f"{x:.4f}")
            summary['test_accuracy'] = summary['test_accuracy'].apply(lambda x: f"{x:.4f}")
            summary['test_rmse'] = summary['test_rmse'].apply(lambda x: f"{x:.4f}")
            summary['training_time_seconds'] = summary['training_time_seconds'].apply(
                lambda x: f"{x:.2f}s"
            )
            
            print(summary.to_string(index=False))
        
        print()
    
    def compare_data_efficiency(
        self,
        dataset: Dataset,
        sample_sizes: List[int] = [50, 100, 200, 500],
        num_trials: int = 3,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare models across different dataset sizes.
        
        This shows which models work better with limited data.
        
        Args:
            dataset: Full dataset to sample from
            sample_sizes: List of student counts to test
            num_trials: Number of trials per size (for variance estimation)
            verbose: Print progress
            
        Returns:
            DataFrame with data efficiency results
        """
        if verbose:
            print("\n" + "="*70)
            print(" DATA EFFICIENCY COMPARISON")
            print("="*70)
            print(f"\nSample sizes: {sample_sizes}")
            print(f"Trials per size: {num_trials}")
            print()
        
        results = []
        
        for size in sample_sizes:
            if size > len(dataset.sequences):
                print(f"Skipping size {size} (exceeds dataset size)")
                continue
            
            if verbose:
                print(f"\nSample size: {size} students")
                print("-" * 70)
            
            for trial in range(num_trials):
                # Sample dataset
                sampled_dataset = self._sample_dataset(dataset, size)
                
                # Split into train/test
                train_data, test_data = self._split_dataset(sampled_dataset, 0.8)
                
                # Evaluate each model
                for model_name, model in self.models.items():
                    try:
                        # Train
                        model.fit(train_data, max_iterations=30, verbose=False)
                        
                        # Evaluate
                        metrics = model.evaluate(test_data)
                        
                        results.append({
                            'sample_size': size,
                            'trial': trial,
                            'model': model_name,
                            'test_auc': metrics['auc'],
                            'test_accuracy': metrics['accuracy'],
                            'test_rmse': metrics['rmse']
                        })
                        
                    except Exception as e:
                        print(f"  Error with {model_name}: {e}")
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "data_efficiency.csv"
        df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\nResults saved to: {output_file}")
        
        self.results['data_efficiency'] = df
        return df
    
    def _sample_dataset(self, dataset: Dataset, num_students: int) -> Dataset:
        """Sample a subset of students from dataset."""
        np.random.seed()  # Different seed each time
        indices = np.random.choice(
            len(dataset.sequences),
            size=min(num_students, len(dataset.sequences)),
            replace=False
        )
        
        return Dataset(
            sequences=[dataset.sequences[i] for i in indices],
            skills=dataset.skills,
            items=dataset.items
        )
    
    def save_summary_report(self) -> None:
        """Save a comprehensive summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(self.models.keys()),
            'experiments_run': list(self.results.keys())
        }
        
        # Add summary statistics
        if 'single_dataset' in self.results:
            df = self.results['single_dataset']
            report['best_model_by_auc'] = df.loc[df['test_auc'].idxmax(), 'model']
            report['best_model_by_accuracy'] = df.loc[df['test_accuracy'].idxmax(), 'model']
            report['fastest_model'] = df.loc[df['training_time_seconds'].idxmin(), 'model']
        
        # Save report
        output_file = self.output_dir / "comparison_summary.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Summary report saved to: {output_file}")


# Convenience function for quick comparison
def quick_model_comparison(
    num_students: int = 200,
    seed: int = 42,
    output_dir: str = "results/quick_comparison"
) -> Dict:
    """
    Run a quick comparison between Standard BKT and BKT with Forgetting.
    
    Args:
        num_students: Number of students in synthetic dataset
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print(" QUICK MODEL COMPARISON")
    print("="*70)
    
    # Generate dataset
    print("\nGenerating dataset...")
    generator = MockDataGenerator(seed=seed)
    dataset = generator.generate_dataset(
        num_students=num_students,
        num_skills=3,
        min_attempts_per_student=30,
        max_attempts_per_student=50,
        forgetting_rate=0.05  # Include some forgetting in data
    )
    
    print(f"Dataset: {dataset.num_students} students, {dataset.num_interactions} interactions")
    
    # Create comparison framework
    comparison = ModelComparison(output_dir=output_dir)
    
    # Add models
    comparison.add_model("Standard BKT", StandardBKT())
    comparison.add_model("BKT with Forgetting", BKTWithForgetting())
    
    # Run comparison
    results_df = comparison.compare_on_dataset(dataset, verbose=True)
    
    # Data efficiency comparison
    print("\nRunning data efficiency analysis...")
    efficiency_df = comparison.compare_data_efficiency(
        dataset,
        sample_sizes=[50, 100, 200],
        num_trials=2,
        verbose=True
    )
    
    # Save summary
    comparison.save_summary_report()
    
    print("\n" + "="*70)
    print(" COMPARISON COMPLETE!")
    print(f" Results saved to: {output_dir}/")
    print("="*70 + "\n")
    
    return {
        'results': results_df,
        'efficiency': efficiency_df,
        'comparison': comparison
    }
