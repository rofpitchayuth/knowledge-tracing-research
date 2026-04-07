"""
save_production_model.py
========================
Production model training script for the DKT-GRU Knowledge Tracing model.

This script:
  1. Loads the full synthetic_data_5000_diverse.csv dataset.
  2. Initialises DeepKnowledgeTracingGRU with the best hyperparameters
     found during the research phase (best_hyperparameters_5000.json).
  3. Trains the model for the configured number of epochs on the ENTIRE
     dataset (no train/val split — we want the model to absorb every
     sequence before production deployment).
  4. Saves two artefacts to data/:
       - dkt_gru_production.pt   → PyTorch state_dict of the GRU nn.Module
       - dkt_gru_skill_map.json  → {original_question_id: integer_index} mapping
                                   (MUST match exactly at inference time)

Usage (run from bkt_experiments/ directory):
    python scripts/save_production_model.py

Requirements: torch, pandas, numpy, tqdm (already in research requirements.txt)
"""

import json
import sys
import time
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make sure all research-project modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # → bkt_experiments/
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader
from models.deep.dkt_gru import DeepKnowledgeTracingGRU

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE        = PROJECT_ROOT / "data" / "synthetic_data_5000_diverse.csv"
HYPERPARAMS_FILE = PROJECT_ROOT / "data" / "best_hyperparameters_5000.json"
OUTPUT_PT        = PROJECT_ROOT / "data" / "dkt_gru_production.pt"
OUTPUT_SKILL_MAP = PROJECT_ROOT / "data" / "dkt_gru_skill_map.json"

# Training epochs — enough to learn patterns without over-engineering the script
TRAIN_EPOCHS = 10

# Device selection (fall back to CPU if CUDA is unavailable)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("  DKT-GRU Production Model Training Script")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load best hyperparameters for the GRU
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading hyperparameters from: {HYPERPARAMS_FILE.name}")
    if not HYPERPARAMS_FILE.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {HYPERPARAMS_FILE}")

    with open(HYPERPARAMS_FILE, "r", encoding="utf-8") as f:
        all_params = json.load(f)

    gru_params = all_params["DeepKnowledgeTracingGRU"]["best_params"]
    hidden_size    = int(gru_params["hidden_size"])       # 64
    dropout        = float(gru_params["dropout"])         # ~0.394
    learning_rate  = float(gru_params["learning_rate"])   # ~0.000949
    batch_size     = int(gru_params["batch_size"])        # 32

    print(f"   hidden_size   = {hidden_size}")
    print(f"   dropout       = {dropout:.4f}")
    print(f"   learning_rate = {learning_rate:.6f}")
    print(f"   batch_size    = {batch_size}")
    print(f"   device        = {DEVICE}")
    print(f"   epochs        = {TRAIN_EPOCHS}")

    # ------------------------------------------------------------------
    # 2. Load the full 5000-student diverse dataset
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading dataset from: {DATA_FILE.name}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

    dataset = DataLoader.load_from_csv(
        filepath=str(DATA_FILE),
        col_student="student_id",
        col_skill="question_id",
        col_item="question_id",
        col_correct="is_correct",
        col_time_taken="response_time",
    )
    print(f"   Students    : {dataset.num_students}")
    print(f"   Interactions: {dataset.num_interactions}")
    print(f"   Unique skills (question_ids): {len(dataset.skills)}")

    # ------------------------------------------------------------------
    # 3. Initialise the GRU model with best hyperparams and train
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training DeepKnowledgeTracingGRU for {TRAIN_EPOCHS} epochs...")
    print(f"   NOTE: Training on FULL dataset (no held-out split) for production.")

    model = DeepKnowledgeTracingGRU(
        hidden_size=hidden_size,
        dropout=dropout,
        device=DEVICE,
    )

    start_time = time.time()
    model.fit(
        dataset=dataset,
        epochs=TRAIN_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=True,           # print epoch-by-epoch progress
    )
    elapsed = time.time() - start_time
    print(f"\n   Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Sanity-check: model must be marked as fitted
    assert model.is_fitted, "model.is_fitted is False — training did not complete!"
    assert model.model is not None, "model.model (nn.Module) is None after training!"

    # ------------------------------------------------------------------
    # 4. Save artefacts
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving production artefacts...")

    # 4a. PyTorch state_dict of the inner nn.Module
    torch.save(model.model.state_dict(), OUTPUT_PT)
    print(f"   ✔  Saved model weights → {OUTPUT_PT}")

    # 4b. skill_map — maps original question_id (as string) → integer index
    #     The DataLoader already converts keys to str, so this is directly usable.
    #     We ALSO save the reverse map and metadata for easy loading in the API.
    skill_map_export = {
        "skill_map": model.skill_map,           # {"0": 0, "1": 1, ...}
        "num_skills": model.num_skills,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "model_arch": "DKTGRUModel",
        "trained_on": str(DATA_FILE.name),
        "epochs": TRAIN_EPOCHS,
    }
    with open(OUTPUT_SKILL_MAP, "w", encoding="utf-8") as f:
        json.dump(skill_map_export, f, indent=2)
    print(f"   ✔  Saved skill map       → {OUTPUT_SKILL_MAP}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUCCESS — Production model artefacts saved:")
    print(f"    {OUTPUT_PT.name:<35} ({OUTPUT_PT.stat().st_size / 1024:.1f} KB)")
    print(f"    {OUTPUT_SKILL_MAP.name:<35} ({OUTPUT_SKILL_MAP.stat().st_size / 1024:.1f} KB)")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Copy both files into the KT_clarify_student_characteristics/ service folder.")
    print("  2. The FastAPI startup event will load these to reconstruct the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
