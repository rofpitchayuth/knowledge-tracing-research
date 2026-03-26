import argparse
from datetime import datetime
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.model_comparison import ModelComparison
from models.bkt.standard_bkt import StandardBKT
from models.bkt.bkt_forgetting import BKTWithForgetting
from models.bkt.individualized_bkt import IndividualizedBKT
from models.bkt.improved_bkt import ImprovedBKT
from models.logistic.logistic_model import LogisticModel

from models.deep.dkt import DeepKnowledgeTracing
from models.deep.dkt_bi_lstm import DeepKnowledgeTracingBiLSTM
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# Import the DataLoader from your data module
from data.data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(
        description='Compare BKT variants, Logistic Regression and Deep Learning across multiple dataset sizes'
    )
    parser.add_argument(
        '--data-files',
        nargs='+',
        type=str,
        default=[
            'synthetic_data_500.csv', 
            'synthetic_data_1000.csv', 
            'synthetic_data_5000.csv'
        ],
        help='List of CSV files to evaluate (separated by space)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='DKT training epochs (default: 15)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Base output directory (default: results/complete_TIMESTAMP)'
    )
    parser.add_argument(
        '--skip-dkt',
        action='store_true',
        help='Skip DKT (faster, for testing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (auto, cuda, cpu, mps)'
    )
    
    args = parser.parse_args()
    
    # Generate base output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"results/complete_{timestamp}"
    else:
        base_output_dir = args.output
        
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU for Deep Learning models.")
            device = 'cpu'
    
    print(f"Configuration:")
    print(f"  Data Files: {args.data_files}") 
    print(f"  Base Output: {base_output_dir}")
    print(f"  Device: {device}")
    
    if device == 'cpu' and not args.skip_dkt:
        print("\nWARNING: You are training Deep Learning models on CPU.")
        print("This may be very slow. Consider verifying your CUDA installation if you have a GPU.")
        print("Run 'pip install torch --index-url https://download.pytorch.org/whl/cu118' (or relevant version).\n")
    
    print()
    
    # OUTER LOOP: Run the entire evaluation pipeline 3 times (Trial 1, 2, 3)
    for trial in range(1, 4):
        print(f"\n{'='*60}")
        print(f"--- STARTING TRIAL {trial} ---")
        print(f"{'='*60}")

        # INNER LOOP: Iterate through each dataset file for this trial
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
                    col_student='student_id',
                    col_skill='question_id',
                    col_item='question_id',
                    col_correct='is_correct',
                    col_time=None,
                    col_time_taken='response_time'
                )
                print(f"Dataset ready: {dataset.num_students} students, {dataset.num_interactions} interactions")
                
                # 2. Setup Output Directory for this specific file and trial
                file_output_dir = f"{base_output_dir}/trial_{trial}/{data_path.stem}"
                
                # 3. Create comparison framework
                comparison = ModelComparison(output_dir=file_output_dir)
                
                print("Adding Traditional Models (BKT & Logistic)...")
                comparison.add_model("Standard BKT", StandardBKT())
                comparison.add_model("BKT with Forgetting", BKTWithForgetting())
                comparison.add_model("Individualized BKT", IndividualizedBKT())
                comparison.add_model("Improved BKT (Time)", ImprovedBKT())
                comparison.add_model("Logistic Model (PFA)", LogisticModel())
                
                if not args.skip_dkt:
                    print(f"Adding Deep Learning Models (DKT) on device: {device.upper()}...")
                    dkt = DeepKnowledgeTracing(hidden_size=128, num_layers=1, dropout=0.2, device=device)
                    comparison.add_model("Deep Knowledge Tracing (LSTM)", dkt)
                    
                    dkt_bi = DeepKnowledgeTracingBiLSTM(hidden_size=128, num_layers=1, dropout=0.2, device=device)
                    comparison.add_model("Deep Knowledge Tracing (Bi-LSTM)", dkt_bi)
                    
                    dkt_gru = DeepKnowledgeTracingGRU(hidden_size=128, num_layers=1, dropout=0.2, device=device)
                    comparison.add_model("Deep Knowledge Tracing (GRU)", dkt_gru)
                else:
                    print("Skipping Deep Learning Models (--skip-dkt flag is active).")
                
                # 4. Run comparison
                fit_params = {
                    'max_iterations': 50,
                    'verbose': False,
                    'epochs': args.epochs,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
                
                print("RUNNING COMPARISON... (This will evaluate ALL added models)")
                results_df = comparison.compare_on_dataset(
                    dataset,
                    fit_params=fit_params,
                    verbose=True
                )
                
                # Save summary for this specific dataset size
                comparison.save_summary_report()
                
                # 5. Print Findings for this dataset
                best_by_auc = results_df.loc[results_df['test_auc'].idxmax()]
                best_by_speed = results_df.loc[results_df['training_time_seconds'].idxmin()]
                
                print(f"\n KEY FINDINGS FOR {data_file}")
                print(f" Best Accuracy: {best_by_auc['model']} (AUC: {best_by_auc['test_auc']:.4f})")
                print(f" Fastest:       {best_by_speed['model']} (Time: {best_by_speed['training_time_seconds']:.2f}s)")
                print(f" Results saved to: {file_output_dir}/")
                
            except Exception as e:
                print(f"Error processing {data_file}: {str(e)}")
                continue

        print(f"\n--- TRIAL {trial} COMPLETED ---")

    print(f"\n ALL EVALUATIONS COMPLETED. Base output folder: {base_output_dir}")

if __name__ == "__main__":
    main()