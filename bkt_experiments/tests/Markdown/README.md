# BKT Experiments - Research Framework

A comprehensive standalone research framework for analyzing Bayesian Knowledge Tracing (BKT) models and variants. This project is **independent** of the main web application and designed for academic research and experimentation.

## 🎯 Purpose

This framework helps answer critical research questions about BKT models:
- **Parameter Sensitivity**: How does each parameter affect model performance?
- **Optimal Ranges**: What are the best initialization values for each parameter?
- **Extreme Behaviors**: What happens when parameters are too high or too low?
- **Parameter Interactions**: How do parameters interact with each other?
- **Model Comparison**: How do different BKT variants and deep learning models compare?

## 📁 Project Structure

```
bkt_experiments/
├── models/                      # Model implementations
│   ├── base.py                 # Abstract base classes
│   └── bkt/
│       ├── standard_bkt.py     # Classic BKT (4 parameters)
│       ├── bkt_forgetting.py   # BKT + forgetting (TODO)
│       ├── individualized_bkt.py  # Per-student BKT (TODO)
│       └── contextual_bkt.py   # Item difficulty + bias (TODO)
├── data/
│   ├── schemas.py              # Data structures
│   ├── mock_generator.py       # Synthetic data generation
│   └── data_loader.py          # Real data loading (TODO)
├── experiments/
│   ├── parameter_sensitivity.py  # Parameter analysis experiments
│   ├── model_comparison.py     # Model comparison (TODO)
│   └── cross_validation.py     # Cross-validation (TODO)
├── analysis/
│   ├── metrics.py              # Evaluation metrics (AUC, RMSE, etc.)
│   ├── visualizations.py       # Publication-quality plots
│   └── statistics.py           # Statistical tests (TODO)
├── notebooks/                   # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_parameter_exploration.ipynb  (TODO)
│   └── 03_model_comparison.ipynb       (TODO)
├── results/                     # Generated outputs
│   ├── figures/
│   ├── tables/
│   └── reports/
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
   ```bash
   cd backend/bkt_experiments
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```python
   from models.bkt.standard_bkt import StandardBKT
   from data.mock_generator import MockDataGenerator
   
   # Generate sample data
   generator = MockDataGenerator(seed=42)
   dataset = generator.generate_dataset(num_students=50)
   
   # Create and fit model
   model = StandardBKT()
   model.fit(dataset, verbose=True)
   
   # Evaluate
   metrics = model.evaluate(dataset)
   print(f"AUC: {metrics['auc']:.4f}")
   ```

### Quick Experiments

#### 1. Parameter Sensitivity Analysis

Run comprehensive parameter sweep:

```python
from experiments.parameter_sensitivity import quick_sensitivity_analysis

results = quick_sensitivity_analysis(
    num_students=200,
    seed=42,
    output_dir="results/my_experiment"
)

# Results include:
# - Parameter sweeps for P(L0), P(T), P(G), P(S)
# - Parameter interaction heatmaps
# - Extreme value analysis
# - Recommendations for initialization
```

Outputs saved to `results/my_experiment/`:
- `sweep_p_init.csv`, `sweep_p_learn.csv`, etc.
- `interaction_p_init_vs_p_learn.csv`
- `extreme_values_analysis.json`
- `parameter_recommendations.json`

#### 2. Visualize Results

```python
from analysis.visualizations import plot_all_parameter_sensitivities
import pandas as pd

# Load sweep results
sweeps = {
    'p_init': pd.read_csv('results/my_experiment/sweep_p_init.csv'),
    'p_learn': pd.read_csv('results/my_experiment/sweep_p_learn.csv'),
    'p_guess': pd.read_csv('results/my_experiment/sweep_p_guess.csv'),
    'p_slip': pd.read_csv('results/my_experiment/sweep_p_slip.csv'),
}

# Create publication-quality figure
plot_all_parameter_sensitivities(sweeps, save_dir="results/figures")
```

## 📊 Key Features

### 1. Mock Data Generation

Generate realistic student learning data with diverse learner profiles:

```python
from data.mock_generator import MockDataGenerator, LearnerProfile

generator = MockDataGenerator(seed=42)

# General dataset
dataset = generator.generate_dataset(
    num_students=100,
    num_skills=5,
    skill_names=["Limits", "Derivatives", "Integrals", "Chain Rule", "U-Substitution"],
    min_attempts_per_student=20,
    max_attempts_per_student=100
)

# Ground-truth dataset (for parameter recovery experiments)
true_params = {
    'skill_00': {'p_init': 0.25, 'p_learn': 0.18, 'p_guess': 0.15, 'p_slip': 0.08},
    'skill_01': {'p_init': 0.15, 'p_learn': 0.25, 'p_guess': 0.20, 'p_slip': 0.12}
}
dataset = generator.generate_simple_dataset(true_params, num_students=100)
```

**Learner Profiles**:
- **Fast Learner**: Low init, high learn rate, low slip/guess
- **Slow Learner**: Low init, low learn rate, high slip/guess
- **Prior Knowledge**: High init, medium learn rate
- **Struggling**: Very low init, low learn, high guess/slip
- **Average**: Balanced parameters

### 2. Standard BKT Implementation

Clean, research-grade implementation with:
- ✅ 4 parameters: P(L0), P(T), P(G), P(S)
- ✅ Bayesian update equations
- ✅ EM algorithm for parameter estimation
- ✅ Numerical stability (log-space computations)
- ✅ Parameter validation and constraints

```python
from models.bkt.standard_bkt import StandardBKT, BKTParameters

# Create model with custom parameters
model = StandardBKT()
model.set_parameters('skill_00', {
    'p_init': 0.20,
    'p_learn': 0.15,
    'p_guess': 0.18,
    'p_slip': 0.10
})

# Or fit from data
model.fit(dataset, max_iterations=100, tolerance=1e-4, verbose=True)

# Get learned parameters
params = model.get_parameters('skill_00')
print(params)  # {'p_init': ..., 'p_learn': ..., ...}
```

### 3. Evaluation Metrics

Comprehensive metrics for KT research:

```python
from analysis.metrics import compute_all_metrics, print_metric_summary

metrics = model.evaluate(dataset)
print_metric_summary(metrics, title="Standard BKT Performance")
```

**Metrics Included**:
- **Accuracy Metrics**: AUC, Accuracy, Precision, Recall, F1-score
- **Error Metrics**: RMSE, Log-loss
- **Calibration**: Expected Calibration Error (ECE)
- **Likelihood**: Log-likelihood
- **Confusion Matrix**: TP, TN, FP, FN

### 4. Parameter Sensitivity Analysis

Systematic experiments to understand parameter impact:

```python
from experiments.parameter_sensitivity import ParameterSensitivityExperiment

exp = ParameterSensitivityExperiment(output_dir="results/sensitivity")

# Single parameter sweep
df = exp.run_single_parameter_sweep(
    parameter_name='p_learn',
    param_range=np.linspace(0.0, 0.5, 20),
    dataset=dataset
)

# All parameters
all_sweeps = exp.run_all_parameter_sweeps(dataset)

# Parameter interactions
interaction_df = exp.run_parameter_interaction_analysis(
    'p_init', 'p_learn',
    param1_range=np.linspace(0, 0.8, 10),
    param2_range=np.linspace(0.05, 0.4, 10),
    dataset=dataset
)

# Extreme values
extreme_results = exp.analyze_extreme_values(dataset)

# Get recommendations
recommendations = exp.generate_recommendations(all_sweeps)
```

### 5. Visualization Tools

Create publication-ready figures:

```python
from analysis.visualizations import (
    plot_parameter_sensitivity,
    plot_all_parameter_sensitivities,
    plot_parameter_interaction_heatmap,
    create_summary_report_figure
)

# Single parameter plot
plot_parameter_sensitivity(
    df=sweep_df,
    parameter_name='p_learn',
    metrics=['auc', 'accuracy', 'rmse'],
    save_path='results/figures/p_learn_sensitivity.png'
)

# All parameters in one figure
plot_all_parameter_sensitivities(all_sweeps, save_dir='results/figures')

# Interaction heatmap
plot_parameter_interaction_heatmap(
    df=interaction_df,
    param1_name='p_init',
    param2_name='p_learn',
    metric='auc',
    save_path='results/figures/init_vs_learn_heatmap.png'
)

# Summary recommendations
create_summary_report_figure(recommendations)
```

## 📖 Research Questions Answered

### 1. Parameter Importance
**Question**: Which parameters matter most?

**How to find out**:
```python
results = quick_sensitivity_analysis(num_students=500)
recs = results['recommendations']

for param, info in recs.items():
    print(f"{param}: Sensitivity = {info['sensitivity']}, "
          f"Range width = {info['range_width']:.3f}")
```

**Expected insights**:
- `p_learn` typically has high impact (wide good range = low sensitivity)
- `p_init` matters less once learning starts (high sensitivity)
- `p_guess` and `p_slip` are often correlated

### 2. Optimal Initialization

**Question**: What values should I use to initialize BKT?

**How to find out**:
```python
recs = results['recommendations']
for param, info in recs.items():
    print(f"{param}:")
    print(f"  Optimal: {info['optimal_value']:.3f}")
    print(f"  Safe range: [{info['recommended_range'][0]:.3f}, "
          f"{info['recommended_range'][1]:.3f}]")
```

**Typical recommendations** (based on literature):
- P(L0): 0.10 - 0.25 (most students start with low knowledge)
- P(T): 0.10 - 0.20 (learning happens gradually)
- P(G): 0.15 - 0.25 (some guessing always occurs)
- P(S): 0.05 - 0.15 (occasional errors even when knowing)

### 3. What If Parameters Are Wrong?

**Question**: What happens if P(L0) is too high? Or P(T) too low?

**How to find out**:
```python
extreme_results = exp.analyze_extreme_values(dataset, verbose=True)

for scenario, result in extreme_results.items():
    if result['success']:
        print(f"{scenario}: AUC = {result['metrics']['auc']:.4f}")
    else:
        print(f"{scenario}: FAILED - {result['error']}")
```

**Common findings**:
- **High P(L0)**: Model assumes everyone knows everything → poor learning detection
- **Low P(T)**: Model thinks learning never happens → stuck at prior
- **High P(G) + P(S)**: Violates identifiability constraint → unstable
- **P(T) = 0**: No learning model → constant predictions

### 4. Parameter Interactions

**Question**: Are P(G) and P(S) redundant? Can high P(L0) compensate for low P(T)?

**How to find out**:
```python
from analysis.visualizations import plot_parameter_interaction_heatmap

# Check P(G) vs P(S)
df = exp.run_parameter_interaction_analysis(
    'p_guess', 'p_slip',
    param1_range=np.linspace(0, 0.35, 10),
    param2_range=np.linspace(0, 0.35, 10),
    dataset=dataset
)

plot_parameter_interaction_heatmap(df, 'p_guess', 'p_slip', metric='auc')
```

**Expected patterns**:
- Diagonal region of low performance (high P(G) + high P(S) → identifiability issues)
- P(L0) and P(T) can partially compensate (high init + low learn ≈ low init + high learn)

## 🔬 Example Research Workflow

```python
# 1. Generate data
from data.mock_generator import MockDataGenerator

generator = MockDataGenerator(seed=42)
train_data = generator.generate_dataset(num_students=300, num_skills=5)
test_data = generator.generate_dataset(num_students=100, num_skills=5)

# 2. Run parameter sensitivity
from experiments.parameter_sensitivity import ParameterSensitivityExperiment

exp = ParameterSensitivityExperiment(output_dir="results/research_2026")
sweeps = exp.run_all_parameter_sweeps(train_data)
interactions = exp.run_parameter_interaction_analysis(
    'p_init', 'p_learn',
    np.linspace(0, 0.8, 12),
    np.linspace(0.05, 0.4, 12),
    train_data
)
extremes = exp.analyze_extreme_values(train_data)
recs = exp.generate_recommendations(sweeps)

# 3. Visualize
from analysis.visualizations import *

plot_all_parameter_sensitivities(sweeps, save_dir="results/research_2026/figures")
plot_parameter_interaction_heatmap(
    interactions, 'p_init', 'p_learn', 
    save_path="results/research_2026/figures/init_vs_learn.png"
)
create_summary_report_figure(recs, save_path="results/research_2026/recommendations.png")

# 4. Write up findings for advisor discussion!
```

## 📝 LaTeX-Ready Output

All CSV files can be easily converted to LaTeX tables:

```python
import pandas as pd

df = pd.read_csv('results/sweep_p_learn.csv')
latex_table = df[['value', 'auc', 'accuracy', 'rmse']].to_latex(
    index=False,
    float_format='%.4f',
    caption='Parameter Sensitivity: P(T) Learning Rate',
    label='tab:p_learn_sweep'
)
print(latex_table)
```

## 🚧 TODO / Future Work

- [ ] BKT with Forgetting model
- [ ] Individualized BKT (per-student parameters)
- [ ] Contextual BKT (item difficulty + student bias)
- [ ] Deep Knowledge Tracing (DKT) - LSTM
- [ ] Dynamic Key-Value Memory Networks (DKVMN)
- [ ] Self-Attentive KT (SAKT)
- [ ] Model comparison framework
- [ ] Cross-validation experiments
- [ ] Statistical significance tests
- [ ] Integration with real calculus platform data

## 📚 References

1. Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User modeling and user-adapted interaction*, 4(4), 253-278.

2. Pardos, Z. A., & Heffernan, N. T. (2010). Modeling individualization in a bayesian networks implementation of knowledge tracing. *International Conference on User Modeling, Adaptation, and Personalization*, 255-266.

3. Khajah, M., Lindsey, R. V., & Mozer, M. C. (2016). How deep is knowledge tracing? *Proceedings of the 9th International Conference on Educational Data Mining*.

4. Piech, C., et al. (2015). Deep knowledge tracing. *Advances in neural information processing systems*, 28.

## 📧 Contact

For questions about this research framework, please contact [Your Name/Team].

---

**Note**: This is a standalone research project and is **not** integrated with the main web application. All experiments run independently.
