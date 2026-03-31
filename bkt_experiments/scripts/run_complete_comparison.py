import argparse
import json
from datetime import datetime
import sys
import torch
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.model_comparison import ModelComparison
from models.bkt.standard_bkt import StandardBKT, BKTParameters
from models.bkt.bkt_forgetting import BKTWithForgetting
from models.bkt.individualized_bkt import IndividualizedBKT
from models.bkt.improved_bkt import ImprovedBKT
from models.logistic.logistic_model import LogisticModel

from models.deep.dkt import DeepKnowledgeTracing
from models.deep.dkt_bi_lstm import DeepKnowledgeTracingBiLSTM
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# Import the DataLoader from your data module
from data.data_loader import DataLoader


def inject_params_to_model(model, model_name, config_dict, skills):
    if model_name not in config_dict:
        return model
    
    params = config_dict[model_name].get("best_params", {})
    if not params:
        return model

    # 1. กรณี IndividualizedBKT
    if isinstance(model, IndividualizedBKT):
        for skill_id in skills:
            model.set_parameters(skill_id, {
                'p_guess': params.get('p_guess', 0.2),
                'p_slip': params.get('p_slip', 0.1),
                'default_p_init': params.get('p_init', 0.2),
                'default_p_learn': params.get('p_learn', 0.15)
            })
    
    # 2. กรณี ImprovedBKT (Time-based)
    elif isinstance(model, ImprovedBKT):
        for skill_id in skills:
            logit = lambda p: np.log(p / (1 - p)) if 0 < p < 1 else -2.0
            model.params[skill_id] = {
                'p_init': params.get('p_init', 0.5),
                'p_learn': params.get('p_learn', 0.1),
                'w_s': 0.0,
                'b_s': logit(params.get('p_slip', 0.1)),
                'w_g': 0.0,
                'b_g': logit(params.get('p_guess', 0.2))
            }
            model.mean_log_time[skill_id] = 0.0
            model.std_log_time[skill_id] = 1.0

    # 3. กรณี Standard BKT / Forgetting
    elif isinstance(model, (StandardBKT, BKTWithForgetting)):
        try:
            model.params = BKTParameters(
                p_init=params.get('p_init', 0.5),
                p_learn=params.get('p_learn', 0.5),
                p_guess=params.get('p_guess', 0.1),
                p_slip=params.get('p_slip', 0.1)
            )
        except Exception as e:
            print(f"  [WARN] Failed to inject params for {model_name}: {e}")
            
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Compare BKT variants, Logistic Regression and Deep Learning across multiple dataset sizes'
    )
    parser.add_argument('--data-files', nargs='+', type=str, default=['synthetic_data_500.csv', 'synthetic_data_1000.csv', 'synthetic_data_5000.csv'])
    # parser.add_argument('--data-files', nargs='+', type=str, default=['synthetic_data_5000_diverse.csv'])
    parser.add_argument('--config', type=str, default='best_hyperparameters_5000.json')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--skip-dkt', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 1. Load Hyperparameters Config
    tuned_params = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            tuned_params = json.load(f)
        print(f"\nSuccessfully loaded tuned hyperparameters from '{args.config}'")
    else:
        print(f"\nWARNING: Configuration file '{args.config}' not found. Using default parameters.")

    # Generate base output directory
    base_output_dir = args.output or f"results/complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = 'cpu'
    if args.device in ['auto', 'cuda'] and torch.cuda.is_available():
        device = 'cuda'
    elif args.device in ['auto', 'mps'] and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        if args.device == 'cuda':
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
    
    print(f"\nConfiguration:")
    print(f"  Data Files: {args.data_files}") 
    print(f"  Base Output: {base_output_dir}")
    print(f"  Device: {device}")
    
    if device == 'cpu' and not args.skip_dkt:
        print("\nWARNING: You are training Deep Learning models on CPU. This may be very slow.")
    
    # OUTER LOOP: Run the entire evaluation pipeline 3 times
    for trial in range(1, 4):
        print(f"\n{'='*60}\n--- STARTING TRIAL {trial} ---\n{'='*60}")

        for data_file in args.data_files:
            print(f" STARTING EVALUATION FOR: {data_file}")
            data_path = Path(data_file)
            
            if not data_path.exists():
                print(f"Error: Could not find '{data_file}'. Skipping to next file...")
                continue
                
            try:
                # 1. Load Data
                dataset = DataLoader.load_from_csv(
                    filepath=str(data_path),
                    col_student='student_id', col_skill='question_id', col_item='question_id',
                    col_correct='is_correct', col_time_taken='response_time'
                )
                print(f"Dataset ready: {dataset.num_students} students, {dataset.num_interactions} interactions")
                
                # ดึงรายการ skills เพื่อนำไปเตรียมช่องว่าง (Initialize) พารามิเตอร์ให้โมเดล BKT ขั้นสูง
                skills = dataset.get_skill_ids() if hasattr(dataset, 'get_skill_ids') else list(dataset.skills)
                num_skills = dataset.num_skills if hasattr(dataset, 'num_skills') else len(skills)

                # 2. Setup Output Directory
                file_output_dir = f"{base_output_dir}/trial_{trial}/{data_path.stem}"
                comparison = ModelComparison(output_dir=file_output_dir)
                
                print("Injecting parameters and adding Traditional Models...")
                
                comparison.add_model("Standard BKT", inject_params_to_model(StandardBKT(), "StandardBKT", tuned_params, skills))
                comparison.add_model("BKT with Forgetting", inject_params_to_model(BKTWithForgetting(), "BKTWithForgetting", tuned_params, skills))
                comparison.add_model("Individualized BKT", inject_params_to_model(IndividualizedBKT(skills=skills), "IndividualizedBKT", tuned_params, skills))
                comparison.add_model("Improved BKT (Time)", inject_params_to_model(ImprovedBKT(), "ImprovedBKT", tuned_params, skills))
                
                log_p = tuned_params.get("LogisticModel", {}).get("best_params", {})
                comparison.add_model("Logistic Model (PFA)", LogisticModel(C=log_p.get("C", 1.0)) if "C" in log_p else LogisticModel())
                
                global_lr, global_batch = 0.001, 32

                if not args.skip_dkt:
                    print(f"Adding Deep Learning Models (DKT) on device: {device.upper()}...")
                    for name, cls in [("LSTM", DeepKnowledgeTracing), ("Bi-LSTM", DeepKnowledgeTracingBiLSTM), ("GRU", DeepKnowledgeTracingGRU)]:
                        config_name = f"DeepKnowledgeTracing{name}" if name != "LSTM" else "DeepKnowledgeTracing"
                        p = tuned_params.get(config_name, {}).get("best_params", {})
                        
                        comparison.add_model(f"Deep Knowledge Tracing ({name})", cls(
                            num_skills=num_skills,
                            hidden_size=p.get("hidden_size", 128),
                            dropout=p.get("dropout", 0.2),
                            device=device
                        ))
                    
                    # Use GRU's parameters as the baseline for execution
                    gru_p = tuned_params.get("DeepKnowledgeTracingGRU", {}).get("best_params", {})
                    if gru_p:
                        global_lr, global_batch = gru_p.get("learning_rate", 0.001), gru_p.get("batch_size", 32)
                
                # 4. Run comparison
                fit_params = {
                    'max_iterations': 50,
                    'verbose': False,
                    'epochs': args.epochs,
                    'batch_size': global_batch,
                    'learning_rate': global_lr
                }
                
                print("RUNNING COMPARISON... (This will evaluate ALL added models)")
                results_df = comparison.compare_on_dataset(dataset, fit_params=fit_params, verbose=True)
                
                comparison.save_summary_report()
                
                best_by_auc = results_df.loc[results_df['test_auc'].idxmax()]
                best_by_speed = results_df.loc[results_df['training_time_seconds'].idxmin()]
                
                print(f"\n KEY FINDINGS FOR {data_file}")
                print(f" Best Accuracy: {best_by_auc['model']} (AUC: {best_by_auc['test_auc']:.4f})")
                print(f" Fastest:       {best_by_speed['model']} (Time: {best_by_speed['training_time_seconds']:.2f}s)")
                
            except Exception as e:
                print(f"Error processing {data_file}: {str(e)}")
                traceback.print_exc()
                continue

        print(f"\n--- TRIAL {trial} COMPLETED ---")

    print(f"\n ALL EVALUATIONS COMPLETED. Base output folder: {base_output_dir}")

if __name__ == "__main__":
    main()