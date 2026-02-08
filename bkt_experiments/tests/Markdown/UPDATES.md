# BKT Experiments - Updated Guide

## 🎉 NEW Features Added!

### 1️⃣ **BKT with Forgetting** (`models/bkt/bkt_forgetting.py`)

Extension of Standard BKT that models knowledge forgetting:

**Key Differences:**
- **5 parameters** instead of 4: P(L0), P(T), **P(F)**, P(G), P(S)
- **Bidirectional transitions**:
  - 0 → 1 (learning) with probability P(T)
  - 1 → 0 (forgetting) with probability P(F) **← NEW!**
- More realistic for long-term retention
- Better parameter identifiability

**When to use:**
- Spaced practice scenarios
- Long time gaps between practice
- Studying retention over days/weeks
- When "once learned, never forgotten" assumption is unrealistic

**Example usage:**
```python
from models.bkt.bkt_forgetting import BKTWithForgetting

# Create model
model = BKTWithForgetting()

# Fit on data
model.fit(dataset, max_iterations=50, verbose=True)

# Get parameters (now includes P(F))
params = model.get_parameters('skill_00')
print(f"Forgetting rate: {params['p_forget']:.4f}")
```

### 2️⃣ **Model Comparison Framework** (`experiments/model_comparison.py`)

Comprehensive framework for comparing models:

**Features:**
1. **Performance Comparison**
   - Train/test split
   - Multiple metrics (AUC, Accuracy, RMSE, ECE)
   - Overfitting detection
   - Statistical significance (future)

2. **Data Efficiency Analysis**
   - Test models with different sample sizes
   - Shows which model works better with limited data
   - Learning curves

3. **Computational Profiling**
   - Training time
   - Inference time
  - Memory usage (future)

4. **Interpretability vs Accuracy**
   - Compare transparent models (BKT) vs black-box (DKT, future)

**Example usage:**
```python
from experiments.model_comparison import ModelComparison
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting

# Create comparison
comparison = ModelComparison(output_dir="results/my_comparison")

# Add models
comparison.add_model("Standard BKT", StandardBKT())
comparison.add_model("BKT with Forgetting", BKTWithForgetting())

# Compare
results_df = comparison.compare_on_dataset(dataset)

# Data efficiency
efficiency_df = comparison.compare_data_efficiency(
    dataset,
    sample_sizes=[50, 100, 200, 500],
    num_trials=3
)
```

## 🚀 Quick Start: Model Comparison

### Step 1: Run Comparison

```bash
cd backend/bkt_experiments

# Quick comparison (default settings)
python run_model_comparison.py

# Custom settings
python run_model_comparison.py --students 500 --output results/model_comparison
```

**What it does:**
- Generates synthetic dataset
- Trains both Standard BKT and BKT with Forgetting
- Evaluates on test set
- Tests data efficiency (50, 100, 200 students)
- Saves results to CSV/JSON

**Time:** ~5-10 minutes

### Step 2: Create Visualizations

```bash
python create_comparison_plots.py --input results/model_comparison
```

**Outputs:**
- `model_comparison.png` - Bar charts comparing AUC, Accuracy, RMSE
- `data_efficiency.png` - Learning curves showing performance vs sample size

### Step 3: Analyze Results

Check these files:
1. **`comparison_results.csv`** - Main results table
2. **`data_efficiency.csv`** - Performance at different sample sizes
3. **`comparison_summary.json`** - Best models summary

**Key questions answered:**
- Which model is more accurate?
- Which model is faster?
- Which model needs less data?
- Is there overfitting?

## 📊 Understanding Comparison Results

### Example Results

**comparison_results.csv:**
| model | test_auc | test_accuracy | test_rmse | training_time_seconds | accuracy_gap |
|-------|----------|---------------|-----------|----------------------|--------------|
| BKT with Forgetting | 0.7653 | 0.7123 | 0.4521 | 8.45s | 0.032 |
| Standard BKT | 0.7521 | 0.7012 | 0.4678 | 7.23s | 0.045 |

**Interpretation:**
- ✅ **BKT with Forgetting wins** on accuracy (AUC 0.7653 vs 0.7521)
- ⏱️ **Standard BKT faster** (7.23s vs 8.45s)
- 📉 **BKT with Forgetting less overfit** (gap 0.032 vs 0.045)

**Recommendation:** Use BKT with Forgetting if accuracy is priority, Standard BKT if speed matters

### When Forgetting Model Helps

**Forgetting model performs better when:**
1. **Long gaps** between practice sessions
2. **Spaced repetition** data
3. **Multi-day** learning trajectories
4. Real data shows **performance decay** over time

**Standard BKT sufficient when:**
1. **Short timeframes** (single session)
2. **Massed practice** (intensive cramming)
3. **Speed critical** (real-time adaptation)
4. **Interpretability priority** (fewer parameters)

## 🎓 For Academic Research

### Research Questions You Can Answer

1. **Does modeling forgetting improve prediction?**
   → Compare AUC/RMSE of both models

2. **How much does forgetting rate vary by skill?**
   → Look at P(F) across different skills

3. **Trade-off: Accuracy vs Complexity?**
   → 5 parameters (forgetting) justify performance gain?

4. **Data requirements?**
   → Data efficiency plot shows sample size needed

### For Thesis/Paper

**Table 1: Model Comparison**
```python
import pandas as pd

df = pd.read_csv('results/model_comparison/comparison_results.csv')
latex = df[['model', 'test_auc', 'test_accuracy', 'test_rmse', 
            'training_time_seconds']].to_latex(
    index=False,
    float_format='%.4f',
    caption='Comparison of BKT Models on Synthetic Dataset',
    label='tab:model_comparison'
)
print(latex)
```

**Figure 1: Data Efficiency**
```bash
# Already created by create_comparison_plots.py
# File: results/thesis_comparison/data_efficiency.png
```

## 📂 Updated Project Structure

```
bkt_experiments/
├── models/
│   └── bkt/
│       ├── standard_bkt.py          # 4 parameters
│       ├── bkt_forgetting.py        # 5 parameters (NEW!)
│       ├── individualized_bkt.py    # TODO
│       └── contextual_bkt.py        # TODO
│
├── experiments/
│   ├── parameter_sensitivity.py     # ✅ Complete
│   ├── model_comparison.py          # ✅ Complete (NEW!)
│   └── cross_validation.py          # TODO
│
├── run_experiment.py                # Parameter sensitivity
├── run_model_comparison.py          # Model comparison (NEW!)
├── create_visualizations.py         # Sensitivity plots
└── create_comparison_plots.py       # Comparison plots (NEW!)
```

## 🔄 Complete Workflow Example

```bash
# 1. Parameter Sensitivity (understand individual models)
python run_experiment.py --students 300 --output results/exp1
python create_visualizations.py --input results/exp1

# 2. Model Comparison (which model is better?)
python run_model_comparison.py --students 300 --output results/exp2
python create_comparison_plots.py --input results/exp2

# 3. Analyze and write thesis!
```

## 🆕 What's Next?

Still TODO (ordered by priority):

1. **Individualized BKT** - Different parameters per student
2. **Contextual BKT** - Account for item difficulty
3. **Deep Learning Models** - DKT, DKVMN, SAKT
4. **Statistical Tests** - Significance testing
5. **Real Data Integration** - Load from your calculus platform

## 💡 Tips

- Start with **parameter sensitivity** to understand each model
- Then run **model comparison** to choose best model
- Use **forgetting model** if your data has time gaps
- For presentations: Use `comparison_results.csv` → Excel → Pretty table
- For thesis: Use `.to_latex()` for automatic table generation

---

**Questions?** Check:
- `README.md` - Full documentation
- `demo.py` - Basic examples
- `notebooks/01_quick_start.ipynb` - Interactive tutorial
