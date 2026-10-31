# fusion_run.py
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from fusion_train import run_fusion_training
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    with open("/mnt/18T/healmade/vit_patch/MedViT/ViT_2D/configs/base_config.json", "r") as f:
        base_cfg = json.load(f)

    dataset = "organsmnist"  
    img_size = 28
    patch_sizes = [1, 2, 4]
    models = ["vit_small_patch16_224"]
    robustness = 3

    checkpoint_base_dir = "/mnt/18T/healmade/vit_patch/results"
    save_dir = "/mnt/18T/healmade/vit_patch/results"

    cfg = {
        **base_cfg,
        "dataset": dataset,
        "img_size": img_size,
        "patch_sizes": patch_sizes,
        "models": models,
        "robustness": robustness,
        "checkpoint_dir": checkpoint_base_dir,
        "save_dir": save_dir,
        "batch_size": 16,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("Starting Fusion Training...")
    print(f"Checkpoint base directory: {checkpoint_base_dir}")
    print(f"Dataset: {dataset}")
    print(f"Model: {models[0]}")
    print(f"Patch sizes for fusion: {patch_sizes}")
    print("="*60)

   
    agg_df, runs_df = run_fusion_training(cfg, device)

    if agg_df is not None and len(agg_df) > 0:
        print("\n✅ All experiments finished!")
        fusion_save_dir = os.path.join(save_dir, "fusion_results")
        print(f"\nResults saved to: {fusion_save_dir}/")
        print(f"  - fusion_aggregated_results_{dataset}.csv  ")
        print(f"  - fusion_per_run_results_{dataset}.csv  ")
        print(f"  - fusion_paper_table_{dataset}.csv      ")
    else:
        print("\n❌ Fusion training failed. Please check the checkpoint directories and files.")
