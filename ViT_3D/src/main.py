import json
import torch
from src.train import run_experiment

if __name__ == "__main__":
    # --- Load constants from config ---
    with open("../configs/base_config_3d.json", "r") as f:
        base_cfg = json.load(f)

    # --- User inputs ---
    dataset_name = "vesselmnist3d"     # 👈 user sets
    img_size = 28               # 👈 user sets (depends on dataset)
    patch_size = 28             # 👈 user sets: 28, 14, 7, 4, 2, 1
    model_size = "tiny"         # 👈 user sets: tiny, small, base
    robustness = 3              # 👈 user sets

    # --- Merge into one config dict ---
    cfg = {
        **base_cfg,
        "dataset": dataset_name,
        "img_size": img_size,
        "patch_size": patch_size,
        "model_size": model_size,
        "robustness": robustness
    }

    # --- Device selection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Run experiment ---
    results = run_experiment(cfg, device)

    print("\n✅ Experiment finished. Results:")
    for r in results:
        print(r)
