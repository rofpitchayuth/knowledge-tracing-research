"""
tune_hyperparameters.py
=======================
Hyperparameter tuning script for Knowledge Tracing models.

Strategy:
  - BKT models    : Optuna Bayesian optimisation (Fast & Smart searching)
  - Logistic      : Grid Search for Regularization
  - DKT models    : Optuna Bayesian optimisation (Continuous space)

Metric in all cases: Validation AUC (higher is better).

Output: best_hyperparameters.json saved in the working directory.
"""

import sys
import json
import copy
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any

import numpy as np

# ── Allow imports relative to the bkt_experiments package ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from data.schemas import Dataset

# BKT models
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting
from models.bkt.improved_bkt import ImprovedBKT
from models.bkt.individualized_bkt import IndividualizedBKT

# Logistic Model
from models.logistic.logistic_model import LogisticModel

# Deep Learning models
from models.deep.dkt import DeepKnowledgeTracing
from models.deep.dkt_bi_lstm import DeepKnowledgeTracingBiLSTM
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# ── Helpers ────────────────────────────────────────────────────────────────────

def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1):
    sequences = dataset.sequences
    n = len(sequences)

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    def _subset(idx_array):
        seqs = [sequences[i] for i in idx_array]
        from data.schemas import Dataset as DS
        return DS(sequences=seqs, skills=dataset.skills, items=dataset.items)

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def get_val_auc(model, val_dataset: Dataset) -> float:
    try:
        metrics = model.evaluate(val_dataset)
        return float(metrics.get("auc", 0.0))
    except Exception as e:
        print(f"    [WARN] evaluate() failed: {e}")
        return 0.0


# ── BKT Optuna Tuning ──────────────────────────────────────────────────────────

BKT_MODELS = [
    ("StandardBKT",        StandardBKT),
    ("BKTWithForgetting",  BKTWithForgetting),
    ("ImprovedBKT",        ImprovedBKT),
    ("IndividualizedBKT",  IndividualizedBKT),
]

def _make_bkt_model(model_class, params: Dict[str, float]):
    from models.bkt.standard_bkt import BKTParameters
    default = BKTParameters(
        p_init=params["p_init"],
        p_learn=params["p_learn"],
        p_guess=params["p_guess"],
        p_slip=params["p_slip"],
    )
    try:
        return model_class(default_params=default)
    except TypeError:
        return model_class()

def tune_bkt_optuna(train_dataset, val_dataset, model_name, model_class, n_trials=50, max_iterations=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna is required for tuning. Install it with: pip install optuna")

    def objective(trial):
        # 🌟 กำหนด Search Space อย่างสมเหตุสมผลตามหลัก Psychometrics
        p_init = trial.suggest_float("p_init", 0.01, 0.80)
        p_learn = trial.suggest_float("p_learn", 0.01, 0.80)
        p_guess = trial.suggest_float("p_guess", 0.01, 0.30)  # ไม่ควรเดาถูกเกิน 30%
        p_slip = trial.suggest_float("p_slip", 0.01, 0.10)    # ไม่ควรสะเพร่าเกิน 10%

        params = {
            "p_init": p_init,
            "p_learn": p_learn,
            "p_guess": p_guess,
            "p_slip": p_slip
        }

        try:
            model = _make_bkt_model(model_class, params)
            model.fit(train_dataset, max_iterations=max_iterations, verbose=False)
            val_auc = get_val_auc(model, val_dataset)
        except Exception:
            return 0.0

        return val_auc

    print(f"\n  [{model_name}] Optuna study: {n_trials} trials …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"  [{model_name}] Best Val AUC: {best_trial.value:.4f} | Best Params: {best_trial.params}")
    return {"best_params": best_trial.params, "best_val_auc": round(best_trial.value, 6)}


# ── Logistic Model Tuning ──────────────────────────────────────────────────────
def tune_logistic_model(train_dataset, val_dataset):
    print(f"\n{'─'*60}")
    print("  Logistic Model Grid Search")
    print(f"{'─'*60}")
    
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_auc = -1.0
    best_params = {}
    
    for c in C_values:
        try:
            try:
                model = LogisticModel(C=c)
            except TypeError:
                try:
                    model = LogisticModel(l2_penalty=1/c)
                except TypeError:
                    model = LogisticModel()
                    model.fit(train_dataset, verbose=False)
                    best_auc = get_val_auc(model, val_dataset)
                    print("  [LogisticModel] Model does not accept hyperparameters. Using defaults.")
                    return {"best_params": {}, "best_val_auc": round(best_auc, 6)}

            model.fit(train_dataset, verbose=False)
            val_auc = get_val_auc(model, val_dataset)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_params = {"C": c}
        except Exception:
            continue
            
    print(f"  [LogisticModel] Best Val AUC: {best_auc:.4f} | Best Params: {best_params}")
    return {"best_params": best_params, "best_val_auc": round(best_auc, 6)}


# ── DKT Optuna Tuning ──────────────────────────────────────────────────────────

DKT_MODELS = [
    ("DeepKnowledgeTracing",       DeepKnowledgeTracing),
    ("DeepKnowledgeTracingBiLSTM", DeepKnowledgeTracingBiLSTM),
    ("DeepKnowledgeTracingGRU",    DeepKnowledgeTracingGRU),
]

def tune_dkt_optuna(train_dataset, val_dataset, model_name, model_class, num_skills, n_trials=30, epochs=10, device="cpu"):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna is required for DKT tuning. Install it with: pip install optuna")

    def objective(trial):
        lr          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout     = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64])

        try:
            try:
                model = model_class(num_skills=num_skills, hidden_size=hidden_size, dropout=dropout, device=device)
            except TypeError:
                model = model_class(hidden_size=hidden_size, dropout=dropout, device=device)
                
            model.fit(train_dataset, epochs=epochs, batch_size=batch_size, learning_rate=lr, verbose=False)
            val_auc = get_val_auc(model, val_dataset)
        except Exception:
            return 0.0

        return val_auc

    print(f"\n  [{model_name}] Optuna study: {n_trials} trials …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"  [{model_name}] Best Val AUC: {best_trial.value:.4f} | Best Params: {best_trial.params}")
    return {"best_params": best_trial.params, "best_val_auc": round(best_trial.value, 6)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="synthetic_data_1000.csv")
    parser.add_argument("--bkt-trials", type=int, default=50, help="Number of Optuna trials for BKT")
    parser.add_argument("--dkt-trials", type=int, default=20, help="Number of Optuna trials for DKT")
    parser.add_argument("--dkt-epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-dkt", action="store_true")
    parser.add_argument("--skip-bkt", action="store_true")
    parser.add_argument("--bkt-max-iter", type=int, default=50)
    parser.add_argument("--output", type=str, default="best_hyperparameters.json")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    print(f"\n{'='*60}\n  Knowledge Tracing Hyperparameter Tuning\n{'='*60}")
    print("Loading dataset …")
    
    full_dataset = DataLoader.load_from_csv(
        filepath=str(data_path),
        col_student="student_id", col_skill="question_id", col_item="question_id",
        col_correct="is_correct", col_time=None, col_time_taken="response_time",
    )
    
    num_skills = full_dataset.num_skills if hasattr(full_dataset, 'num_skills') else len(full_dataset.skills)

    train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1)
    print(f"\nSplit → Train: {train_ds.num_students} students | Val: {val_ds.num_students} students | Test: {test_ds.num_students} students")

    all_results = {}

    # --- BKT Optuna Search ---
    if not args.skip_bkt:
        print(f"\n{'─'*60}\n  BKT Optuna Optimisation\n{'─'*60}")
        for name, cls in BKT_MODELS:
            all_results[name] = tune_bkt_optuna(
                train_ds, val_ds, name, cls, 
                n_trials=args.bkt_trials, max_iterations=args.bkt_max_iter
            )
        
        # --- Logistic Model ---
        all_results["LogisticModel"] = tune_logistic_model(train_ds, val_ds)

    # --- DKT Optuna ---
    if not args.skip_dkt:
        print(f"\n{'─'*60}\n  DKT Optuna Optimisation\n{'─'*60}")
        for name, cls in DKT_MODELS:
            all_results[name] = tune_dkt_optuna(
                train_ds, val_ds, name, cls, num_skills,
                n_trials=args.dkt_trials, epochs=args.dkt_epochs, device=args.device
            )

    # ── 4. Save results ────────────────────────────────────────────────────────
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}\n  Tuning complete! Results saved to: {output_path}\n{'='*60}\n")
    print(f"{'Model':<35} {'Best Val AUC':>14}\n" + "─" * 51)
    for model_name, res in all_results.items():
        print(f"  {model_name:<33} {res.get('best_val_auc', 0.0):>14.4f}")
    print()

if __name__ == "__main__":
    main()