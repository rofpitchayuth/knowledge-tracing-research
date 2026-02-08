# BKT Experiments - Getting Started

## Quick Installation

1. **Navigate to the directory**:
   ```bash
   cd backend/bkt_experiments
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   python demo.py
   ```

## What You Get

This framework provides everything you need for BKT research:

### ✅ Implemented Features

1. **Standard BKT Model** (`models/bkt/standard_bkt.py`)
   - 4 parameters: P(L0), P(T), P(G), P(S)
   - EM algorithm for parameter estimation
   - Bayesian update equations
   - Numerical stability

2. **Mock Data Generator** (`data/mock_generator.py`)
   - 5 learner profiles (Fast, Slow, Prior Knowledge, Struggling, Average)
   - Realistic learning trajectories
   - Multiple skills and items
   - Temporal effects

3. **Parameter Sensitivity Analysis** (`experiments/parameter_sensitivity.py`)
   - Single parameter sweeps
   - 2D parameter interactions
   - Extreme value testing
   - Automated recommendations

4. **Evaluation Metrics** (`analysis/metrics.py`)
   - AUC, Accuracy, Precision, Recall, F1
   - RMSE, Log-loss
   - Expected Calibration Error (ECE)
   - Confusion matrices

5. **Visualizations** (`analysis/visualizations.py`)
   - Parameter sensitivity curves
   - Interaction heatmaps
   - Learning trajectories
   - Summary reports

6. **Documentation**
   - Comprehensive README
   - Demo script
   - Jupyter notebook tutorial

### 🚧 TODO (Future Work)

- BKT with Forgetting
- Individualized BKT
- Contextual BKT
- Deep Learning models (DKT, DKVMN, SAKT)
- Model comparison framework
- Cross-validation
- Statistical significance tests

## Example Usage

### Quick Experiment

```python
from experiments.parameter_sensitivity import quick_sensitivity_analysis

# Run full parameter analysis (takes ~10 minutes)
results = quick_sensitivity_analysis(
    num_students=200,
    seed=42,
    output_dir="results/experiment_001"
)

# View recommendations
print(results['recommendations'])
```

### Custom Analysis

```python
from data.mock_generator import MockDataGenerator
from models.bkt.standard_bkt import StandardBKT
from experiments.parameter_sensitivity import ParameterSensitivityExperiment
from analysis.visualizations import *

# 1. Generate data
generator = MockDataGenerator(seed=42)
dataset = generator.generate_dataset(num_students=100, num_skills=3)

# 2. Fit model
model = StandardBKT()
model.fit(dataset, verbose=True)

# 3. Analyze parameters
exp = ParameterSensitivityExperiment(output_dir="results/my_research")
sweeps = exp.run_all_parameter_sweeps(dataset)

# 4. Visualize
plot_all_parameter_sensitivities(sweeps, save_dir="results/my_research/figures")
```

## For Your Research

This framework is designed to help you answer:

1. **Which parameters matter most?**
   → Run sensitivity sweeps and check range widths

2. **What values should I use?**
   → Check `parameter_recommendations.json`

3. **How do parameters interact?**
   → Look at interaction heatmaps

4. **What if parameters are wrong?**
   → Review extreme value analysis results

## Next Steps

1. **Run the demo**: `python demo.py`
2. **Open the notebook**: `notebooks/01_quick_start.ipynb`
3. **Read the README**: Full documentation
4. **Run experiments**: Use your own research questions

## Support

- Check README.md for detailed documentation
- Run demo.py for working examples
- Explore notebooks for interactive tutorials

---

**This is a standalone research project, independent of the main web application.**
