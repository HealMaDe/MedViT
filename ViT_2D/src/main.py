import json
import torch
from src.train import run_experiment

if __name__ == "__main__":
    # --- Load constants from config ---
    with open("../configs/base_config.json", "r") as f:
        base_cfg = json.load(f)

    # --- User inputs ---
    dataset = "organsmnist_128"     # 👈 user sets
    img_size = 128               # 👈 user sets (depends on dataset)
    patch_sizes = [1]   # 👈 user sets
    models = ["vit_tiny_patch16_224"]  # 👈 user sets
    robustness = 3              # 👈 user sets

    # --- Merge into one config dict ---
    cfg = {
        **base_cfg,
        "dataset": dataset,
        "img_size": img_size,
        "patch_sizes": patch_sizes,
        "models": models,
        "robustness": robustness
    }

    # --- Device selection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Run experiment ---
    results = run_experiment(cfg, device)

    print("\n✅ Experiment finished. Results:")
    for r in results:
        print(r)


