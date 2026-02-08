"""
Parameter Sensitivity Analysis for BKT Models.

This module performs comprehensive parameter sensitivity experiments to answer:
1. How does each parameter affect model performance?
2. What are appropriate initialization ranges?
3. What happens when parameters are extreme?
4. How do parameters interact with each other?
5. Which parameters are most important?

Key experiments:
- Single parameter sweeps
- Parameter interaction analysis (2D heatmaps)
- Extreme value analysis
- Initialization study
- Convergence analysis
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
import warnings

# Direct imports instead of relative
import sys
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))
from models.bkt.standard_bkt import StandardBKT, BKTParameters
from data.schemas import Dataset
from data.mock_generator import MockDataGenerator
from analysis.metrics import compute_all_metrics


class ParameterSensitivityExperiment:
    """
    Comprehensive parameter sensitivity analysis for BKT models.
    
    This class runs systematic experiments varying BKT parameters to understand:
    - Parameter importance
    - Optimal ranges  
    - Pathological behaviors
    - Parameter interactions
    """
    
    def __init__(self, output_dir: str = "results/parameter_sensitivity"):
        """
        Initialize sensitivity experiment.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.experiment_metadata = {
            'start_time': datetime.now().isoformat(),
            'experiments_run': 0
        }
    
    def run_single_parameter_sweep(
        self,
        parameter_name: str,
        param_range: List[float],
        dataset: Dataset,
        base_params: Optional[BKTParameters] = None,
        num_trials: int = 3,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Sweep a single parameter while keeping others fixed.
        
        This answers: "How important is this parameter? What's its optimal range?"
        
        Args:
            parameter_name: Name of parameter to vary ('p_init', 'p_learn', 'p_guess', 'p_slip')
            param_range: List of values to test
            dataset: Dataset for evaluation
            base_params: Base parameters (others kept fixed)
            num_trials: Number of trials per value (for variance estimation)
            verbose: Print progress
            
        Returns:
            DataFrame with results for each parameter value
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Parameter Sensitivity Sweep: {parameter_name}")
            print(f"{'='*70}")
            print(f"Testing {len(param_range)} values with {num_trials} trials each")
            print(f"Base parameters: {base_params.to_dict() if base_params else 'default'}\n")
        
        if base_params is None:
            base_params = BKTParameters()
        
        results = []
        
        iterator = tqdm(param_range, desc=f"Sweeping {parameter_name}") if verbose else param_range
        
        for param_value in iterator:
            # Create params with this value
            params_dict = base_params.to_dict()
            params_dict[parameter_name] = param_value
            
            # Run multiple trials
            trial_metrics = []
            
            for trial in range(num_trials):
                try:
                    # Create and fit model
                    model = StandardBKT()
                    
                    # Set parameters for all skills
                    for skill_id in dataset.get_skill_ids():
                        model.set_parameters(skill_id, params_dict)
                    
                    model.is_fitted = True  # Skip EM training, use fixed params
                    
                    # Evaluate
                    metrics = model.evaluate(dataset)
                    trial_metrics.append(metrics)
                    
                except Exception as e:
                    if verbose:
                        warnings.warn(f"Failed for {parameter_name}={param_value}: {e}")
                    continue
            
            if trial_metrics:
                # Aggregate across trials
                avg_metrics = {
                    metric: np.mean([m[metric] for m in trial_metrics])
                    for metric in trial_metrics[0].keys()
                }
                std_metrics = {
                    f"{metric}_std": np.std([m[metric] for m in trial_metrics])
                    for metric in trial_metrics[0].keys()
                }
                
                result = {
                    'parameter': parameter_name,
                    'value': param_value,
                    'num_trials': len(trial_metrics),
                    **avg_metrics,
                    **std_metrics
                }
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / f"sweep_{parameter_name}.csv"
        df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\nResults saved to: {output_file}")
            self._print_sweep_summary(df, parameter_name)
        
        self.experiment_metadata['experiments_run'] += 1
        return df
    
    def _print_sweep_summary(self, df: pd.DataFrame, param_name: str):
        """Print summary of sweep results."""
        print(f"\n{'-'*70}")
        print(f"Summary for {param_name.upper()}:")
        print(f"{'-'*70}")
        
        # Find best value by AUC
        best_idx = df['auc'].idxmax()
        best_row = df.iloc[best_idx]
        
        print(f"Best value: {best_row['value']:.3f}")
        print(f"  AUC: {best_row['auc']:.4f} (±{best_row.get('auc_std', 0):.4f})")
        print(f"  Accuracy: {best_row['accuracy']:.4f}")
        print(f"  RMSE: {best_row['rmse']:.4f}")
        
        # Range where AUC > 90% of best
        threshold = 0.9 * best_row['auc']
        good_range = df[df['auc'] >= threshold]
        if len(good_range) > 0:
            print(f"\nGood range (AUC ≥ {threshold:.4f}):")
            print(f"  [{good_range['value'].min():.3f}, {good_range['value'].max():.3f}]")
    
    def run_all_parameter_sweeps(
        self,
        dataset: Dataset,
        p_init_range: Optional[List[float]] = None,
        p_learn_range: Optional[List[float]] = None,
        p_guess_range: Optional[List[float]] = None,
        p_slip_range: Optional[List[float]] = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run sensitivity sweeps for all four BKT parameters.
        
        Args:
            dataset: Dataset for evaluation
            p_init_range: Values to test for P(L0)
            p_learn_range: Values to test for P(T)
            p_guess_range: Values to test for P(G)
            p_slip_range: Values to test for P(S)
            verbose: Print progress
            
        Returns:
            Dictionary mapping parameter names to result DataFrames
        """
        # Default ranges based on literature
        if p_init_range is None:
            p_init_range = np.linspace(0.0, 0.9, 10).tolist()
        if p_learn_range is None:
            p_learn_range = np.linspace(0.01, 0.50, 15).tolist()
        if p_guess_range is None:
            p_guess_range = np.linspace(0.0, 0.40, 9).tolist()
        if p_slip_range is None:
            p_slip_range = np.linspace(0.0, 0.40, 9).tolist()
        
        results = {}
        
        # Base parameters (values to keep fixed while sweeping others)
        base_params = BKTParameters(
            p_init=0.20,
            p_learn=0.15,
            p_guess=0.20,
            p_slip=0.10
        )
        
        # Sweep each parameter
        results['p_init'] = self.run_single_parameter_sweep(
            'p_init', p_init_range, dataset, base_params, verbose=verbose
        )
        
        results['p_learn'] = self.run_single_parameter_sweep(
            'p_learn', p_learn_range, dataset, base_params, verbose=verbose
        )
        
        results['p_guess'] = self.run_single_parameter_sweep(
            'p_guess', p_guess_range, dataset, base_params, verbose=verbose
        )
        
        results['p_slip'] = self.run_single_parameter_sweep(
            'p_slip', p_slip_range, dataset, base_params, verbose=verbose
        )
        
        return results
    
    def run_parameter_interaction_analysis(
        self,
        param1_name: str,
        param2_name: str,
        param1_range: List[float],
        param2_range: List[float],
        dataset: Dataset,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Analyze interaction between two parameters using 2D grid search.
        
        This answers: "Do these parameters interact? Are they redundant?"
        
        Args:
            param1_name: First parameter name
            param2_name: Second parameter name
            param1_range: Values for first parameter
            param2_range: Values for second parameter
            dataset: Dataset for evaluation
            verbose: Print progress
            
        Returns:
            DataFrame with results for each parameter combination
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Parameter Interaction Analysis: {param1_name} × {param2_name}")
            print(f"{'='*70}")
            print(f"Testing {len(param1_range)} × {len(param2_range)} = "
                  f"{len(param1_range) * len(param2_range)} combinations\n")
        
        results = []
        base_params = BKTParameters()
        
        total = len(param1_range) * len(param2_range)
        iterator = tqdm(total=total, desc="Grid search") if verbose else None
        
        for val1 in param1_range:
            for val2 in param2_range:
                try:
                    # Create parameters
                    params_dict = base_params.to_dict()
                    params_dict[param1_name] = val1
                    params_dict[param2_name] = val2
                    
                    # Validate constraints
                    if params_dict['p_guess'] + params_dict['p_slip'] >= 0.99:
                        # Skip invalid combinations
                        if iterator:
                            iterator.update(1)
                        continue
                    
                    # Create and evaluate model
                    model = StandardBKT()
                    for skill_id in dataset.get_skill_ids():
                        model.set_parameters(skill_id, params_dict)
                    model.is_fitted = True
                    
                    metrics = model.evaluate(dataset)
                    
                    result = {
                        param1_name: val1,
                        param2_name: val2,
                        **metrics
                    }
                    results.append(result)
                    
                except Exception as e:
                    pass
                
                if iterator:
                    iterator.update(1)
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / f"interaction_{param1_name}_vs_{param2_name}.csv"
        df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\nResults saved to: {output_file}")
        
        self.experiment_metadata['experiments_run'] += 1
        return df
    
    def analyze_extreme_values(
        self,
        dataset: Dataset,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Test extreme parameter values to understand pathological behaviors.
        
        This answers: "What happens when parameters are very high or very low?"
        
        Args:
            dataset: Dataset for evaluation
            verbose: Print progress
            
        Returns:
            Dictionary with extreme value analysis results
        """
        if verbose:
            print(f"\n{'='*70}")
            print("Extreme Value Analysis")
            print(f"{'='*70}\n")
        
        extreme_scenarios = {
            'all_low': BKTParameters(p_init=0.01, p_learn=0.01, p_guess=0.01, p_slip=0.01),
            'all_high': BKTParameters(p_init=0.90, p_learn=0.40, p_guess=0.40, p_slip=0.40),
            'high_init_only': BKTParameters(p_init=0.90, p_learn=0.10, p_guess=0.15, p_slip=0.10),
            'high_learn_only': BKTParameters(p_init=0.20, p_learn=0.45, p_guess=0.15, p_slip=0.10),
            'high_guess_slip': BKTParameters(p_init=0.20, p_learn=0.15, p_guess=0.38, p_slip=0.38),
            'no_learning': BKTParameters(p_init=0.20, p_learn=0.00, p_guess=0.20, p_slip=0.10),
            'perfect_learner': BKTParameters(p_init=0.10, p_learn=0.50, p_guess=0.05, p_slip=0.02),
        }
        
        results = {}
        
        for scenario_name, params in extreme_scenarios.items():
            if verbose:
                print(f"Testing: {scenario_name}")
                print(f"  Params: {params.to_dict()}")
            
            try:
                model = StandardBKT()
                for skill_id in dataset.get_skill_ids():
                    model.set_parameters(skill_id, params.to_dict())
                model.is_fitted = True
                
                metrics = model.evaluate(dataset)
                
                results[scenario_name] = {
                    'parameters': params.to_dict(),
                    'metrics': metrics,
                    'success': True
                }
                
                if verbose:
                    print(f"  AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                results[scenario_name] = {
                    'parameters': params.to_dict(),
                    'error': str(e),
                    'success': False
                }
                if verbose:
                    print(f"  ERROR: {e}")
            
            if verbose:
                print()
        
        # Save results
        output_file = self.output_dir / "extreme_values_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if verbose:
            print(f"Results saved to: {output_file}")
        
        self.experiment_metadata['experiments_run'] += 1
        return results
    
    def generate_recommendations(
        self,
        sweep_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Generate parameter initialization recommendations based on sweep results.
        
        Args:
            sweep_results: Results from run_all_parameter_sweeps
            
        Returns:
            Dictionary with recommendations for each parameter
        """
        recommendations = {}
        
        for param_name, df in sweep_results.items():
            # Find optimal value
            best_idx = df['auc'].idxmax()
            best_value = df.iloc[best_idx]['value']
            best_auc = df.iloc[best_idx]['auc']
            
            # Find reasonable range (within 95% of best AUC)
            threshold = 0.95 * best_auc
            good_values = df[df['auc'] >= threshold]['value']
            
            recommendations[param_name] = {
                'optimal_value': float(best_value),
                'best_auc': float(best_auc),
                'recommended_range': [float(good_values.min()), float(good_values.max())],
                'range_width': float(good_values.max() - good_values.min()),
                'sensitivity': 'high' if (good_values.max() - good_values.min()) < 0.2 else 'low'
            }
        
        # Save recommendations
        output_file = self.output_dir / "parameter_recommendations.json"
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations
    
    def save_experiment_summary(self):
        """Save summary of all experiments run."""
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)


# Convenience function for quick sensitivity analysis
def quick_sensitivity_analysis(
    num_students: int = 200,
    seed: int = 42,
    output_dir: str = "results/quick_sensitivity"
) -> Dict:
    """
    Run a quick parameter sensitivity analysis with default settings.
    
    Args:
        num_students: Number of students in synthetic dataset
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with all results
    """
    print("Generating synthetic dataset...")
    generator = MockDataGenerator(seed=seed)
    dataset = generator.generate_dataset(
        num_students=num_students,
        num_skills=3,
        min_attempts_per_student=30,
        max_attempts_per_student=50
    )
    
    print(f"Dataset created: {dataset.num_students} students, "
          f"{dataset.num_interactions} interactions")
    
    # Run experiments
    exp = ParameterSensitivityExperiment(output_dir=output_dir)
    
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    # 1. Single parameter sweeps
    sweep_results = exp.run_all_parameter_sweeps(dataset)
    
    # 2. Parameter interactions
    interactions = {}
    interactions['init_vs_learn'] = exp.run_parameter_interaction_analysis(
        'p_init', 'p_learn',
        np.linspace(0, 0.8, 8).tolist(),
        np.linspace(0.05, 0.40, 8).tolist(),
        dataset
    )
    
    interactions['guess_vs_slip'] = exp.run_parameter_interaction_analysis(
        'p_guess', 'p_slip',
        np.linspace(0, 0.35, 8).tolist(),
        np.linspace(0, 0.35, 8).tolist(),
        dataset
    )
    
    # 3. Extreme values
    extreme_results = exp.analyze_extreme_values(dataset)
    
    # 4. Generate recommendations
    recommendations = exp.generate_recommendations(sweep_results)
    
    # Save summary
    exp.save_experiment_summary()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return {
        'sweeps': sweep_results,
        'interactions': interactions,
        'extreme_values': extreme_results,
        'recommendations': recommendations
    }
