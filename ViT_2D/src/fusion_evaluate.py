
import os
import re
import glob
import torch
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from data_loader import MedMNISTDataset, get_transforms
from model import get_model
from fusion_model import EarlyFusionModel, CrossAttentionFusion


def find_checkpoint_dirs(base_dir, dataset, model_name, patch_sizes):
    """
    Find checkpoint directories based on the folder structure
    Structure: {base_dir}/organsmnist_28_{model_short}_p{patch_size}/
    """
    checkpoint_dirs = {}
    
    # Extract model short name from full model name
    model_short = model_name.split('_patch')[0] if '_patch' in model_name else model_name
    
    # Try both with and without dataset size suffix
    dataset_variants = [
        dataset,  # "organsmnist"
        f"{dataset}_28",  # "organsmnist_28"
        f"{dataset}_128",  # "organsmnist_128"
    ]
    
    for patch_size in patch_sizes:
        found = False
        for dataset_var in dataset_variants:
            folder_name = f"{dataset_var}_{model_short}_p{patch_size}"
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                checkpoint_dirs[patch_size] = folder_path
                print(f"  ✓ Found checkpoint directory for patch{patch_size}: {folder_path}")
                
                # List available checkpoints for debugging
                files = os.listdir(folder_path)
                pth_files = [f for f in files if f.endswith('.pth')]
                if pth_files:
                    print(f"    Available checkpoints: {', '.join(pth_files[:3])}")
                found = True
                break
        
        if not found:
            print(f"  ✗ Warning: Directory not found for patch{patch_size}")
            # Try to find any matching directory
            search_pattern = os.path.join(base_dir, f"*_{model_short}_p{patch_size}")
            matching_dirs = glob.glob(search_pattern)
            if matching_dirs:
                print(f"    Did you mean one of these? {', '.join(matching_dirs)}")
    
    return checkpoint_dirs


def load_single_model(checkpoint_dir, dataset, model_name, patch_size, 
                      img_size, num_classes, run_id, device):
    """
    Load a single model from checkpoint with memory optimization
    Returns: model or None
    """
    folder_path = checkpoint_dir
    
    # Try to find checkpoint file
    pattern = f"*{model_name}_p{patch_size}_run{run_id}.pth"
    matching_files = glob.glob(os.path.join(folder_path, pattern))
    
    checkpoint_path = None
    if matching_files:
        checkpoint_path = matching_files[0]
        print(f"  Found matching file: {os.path.basename(checkpoint_path)}")
    else:
        # Try explicit patterns with different dataset naming
        dataset_variants = [
            dataset,
            f"{dataset}_28",
            f"{dataset}_128",
        ]
        
        for dataset_var in dataset_variants:
            # Pattern: best_{dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth
            test_path = os.path.join(folder_path, f"best_{dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth")
            if os.path.exists(test_path):
                checkpoint_path = test_path
                print(f"  Found: best_{dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth")
                break
            
            # Pattern: {dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth (without best_)
            test_path = os.path.join(folder_path, f"{dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth")
            if os.path.exists(test_path):
                checkpoint_path = test_path
                print(f"  Found: {dataset_var}_{model_name}_p{patch_size}_run{run_id}.pth")
                break
    
    if checkpoint_path is None:
        print(f"  ✗ Checkpoint not found for patch{patch_size}, run{run_id}")
        return None
    
    try:
        # Load model on CPU first to save GPU memory
        model, _ = get_model(
            model_name, 
            num_classes, 
            patch_size, 
            img_size, 
            "cpu"  # Load on CPU first
        )
        
        # Load checkpoint on CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        # Move to GPU
        model.to(device)
        model.eval()
        
        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"  ✓ Successfully loaded patch{patch_size}, run{run_id}")
        return model
        
    except Exception as e:
        print(f"  ✗ Error loading checkpoint {checkpoint_path}: {e}")
        return None


def evaluate_single_model(model, test_loader, num_classes, device):
    """
    Evaluate a single model on test set
    """
    model.eval()
    all_labels = []
    all_preds_class = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds_class = outputs.argmax(1).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds_class.extend(preds_class.tolist())
            
            if num_classes == 2:
                all_probs.extend(probs[:, 1].tolist())
            else:
                all_probs.extend(probs.tolist())
    
    # Calculate metrics
    acc = np.mean(np.array(all_labels) == np.array(all_preds_class))
    bal_acc = balanced_accuracy_score(all_labels, all_preds_class)
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs))
        else:
            all_probs_array = np.array(all_probs)
            if all_probs_array.shape[1] == num_classes:
                auc = roc_auc_score(all_labels, all_probs_array, multi_class="ovr", average="macro")
            else:
                auc = 0.0
    except Exception as e:
        print(f"AUC Error: {e}")
        auc = 0.0
    
    return acc, bal_acc, auc




def evaluate_fusion_model(model, test_loader, num_classes, device):
    """
    Evaluate a fusion model on test set with proper probability handling
    """
    model.eval()
    all_labels = []
    all_preds_class = []
    all_probs = []
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            
            # Get model outputs
            outputs = model(imgs)
            
            # Apply softmax to get probabilities (works for both binary and multi-class)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            preds_class = probs.argmax(dim=1)
            
            # Convert to numpy
            probs_np = probs.cpu().numpy()
            preds_np = preds_class.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_labels.extend(labels_np.tolist())
            all_preds_class.extend(preds_np.tolist())
            
            # For AUC: handle binary vs multi-class differently
            if num_classes == 2:
                # Binary: use probability of class 1
                all_probs.extend(probs_np[:, 1].tolist())
            else:
                # Multi-class: store all probabilities
                all_probs.extend(probs_np.tolist())
            
            # Clear memory
            del imgs, labels, outputs, probs, preds_class
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds_class = np.array(all_preds_class)
    
    # Calculate metrics
    acc = np.mean(all_labels == all_preds_class)
    bal_acc = balanced_accuracy_score(all_labels, all_preds_class)
    
    # Calculate AUC with proper handling
    try:
        if num_classes == 2:
            # Binary classification
            all_probs = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs)
        else:
            # Multi-class classification
            all_probs = np.array(all_probs)
            
            # Check if probabilities sum to 1
            row_sums = all_probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-5):
                print(f"  Warning: Probabilities don't sum to 1. Row sums: {row_sums[:5]}")
                # Normalize if needed
                all_probs = all_probs / row_sums.reshape(-1, 1)
            
            # Check shape
            if all_probs.shape[1] == num_classes:
                auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
            else:
                print(f"  Error: Probability shape {all_probs.shape} doesn't match num_classes {num_classes}")
                auc = 0.0
                
    except Exception as e:
        print(f"  AUC Error: {e}")
        print(f"  all_labels shape: {all_labels.shape}")
        print(f"  all_probs shape: {all_probs.shape if isinstance(all_probs, np.ndarray) else 'not array'}")
        auc = 0.0
    
    return acc, bal_acc, auc



def run_fusion_experiment(cfg, device):

    dataset = cfg["dataset"]
    img_size = cfg["img_size"]
    patch_sizes = cfg["patch_sizes"]
    models = cfg["models"]
    batch_size = cfg.get("batch_size", 16)  
    save_dir = cfg["save_dir"]
    checkpoint_base_dir = cfg.get("checkpoint_dir", save_dir)
    robustness = cfg.get("robustness", 3)
    
    # Create fusion results directory
    fusion_save_dir = os.path.join(save_dir, "fusion_results")
    os.makedirs(fusion_save_dir, exist_ok=True)
    
    # Load test dataset
    npz_path = f"/mnt/18T/healmade/vit_patch/MedViT/ViT_2D/data/{dataset}.npz"
    _, val_tf = get_transforms(img_size)
    test_ds = MedMNISTDataset("test", dataset, npz_path, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(set(test_ds.labels.flatten()))
    model_name = models[0]
    
    # Find checkpoint directories for each patch size
    print("\n" + "="*60)
    print("Finding checkpoint directories...")
    print(f"Base directory: {checkpoint_base_dir}")
    print("="*60)
    
    checkpoint_dirs = find_checkpoint_dirs(
        checkpoint_base_dir, dataset, model_name, patch_sizes
    )
    
    if not checkpoint_dirs:
        print("\n❌ Error: No checkpoint directories found!")
        print("Expected folder pattern examples:")
        print(f"  - {checkpoint_base_dir}/organsmnist_28_vit_tiny_p1")
        print(f"  - {checkpoint_base_dir}/organsmnist_28_vit_tiny_p2")
        print(f"  - {checkpoint_base_dir}/organsmnist_28_vit_tiny_p4")
        return None, None
    
    print(f"\nFound checkpoint directories for {len(checkpoint_dirs)} patch sizes")
    
    all_results = []
    
    print("\n" + "="*60)
    print("Fusion Experiments with All Runs (Memory Optimized)")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"Number of runs: {robustness}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    # 1. Baseline: Individual models (all runs)
    print("\n" + "="*50)
    print("1. Evaluating individual models (baseline) for all runs...")
    print("="*50)
    
    for run_id in range(1, robustness + 1):
        for patch_size in patch_sizes:
            print(f"\nRun {run_id}, patch_size={patch_size}")
            
            # Clear GPU cache before loading new model
            torch.cuda.empty_cache()
            
            # Load single model
            model = load_single_model(
                checkpoint_dirs[patch_size], dataset, model_name, patch_size, 
                img_size, num_classes, run_id, device
            )
            
            if model is None:
                print(f"  Skipping: No model loaded for run{run_id}, patch{patch_size}")
                continue
                
            acc, bal_acc, auc = evaluate_single_model(
                model, test_loader, num_classes, device
            )
            
            all_results.append({
                "dataset": dataset,
                "fusion_method": "baseline",
                "patch_size": patch_size,
                "run_id": run_id,
                "model_name": model_name,
                "test_acc": round(acc*100, 2),
                "test_bal_acc": round(bal_acc*100, 2),
                "test_auc": round(auc*100, 2)
            })
            
            print(f"  Results: Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, AUC={auc*100:.2f}%")
            
            # Delete model and clear cache
            del model
            torch.cuda.empty_cache()
    
    # Check if we have any results before proceeding
    if not all_results:
        print("\n❌ No models could be loaded. Please check:")
        print("  1. The checkpoint files exist in the directories")
        print("  2. The filename pattern matches")
        print("  3. The model names are correct")
        return None, None
    
    # 2. Same-run fusion (run1, run2, run3)
    print("\n" + "="*50)
    print("2. Evaluating same-run fusion (one model at a time)...")
    print("="*50)
    
    for run_id in range(1, robustness + 1):
        print(f"\nRun {run_id} - using models from same run")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Load models one by one and store in dict
        models_dict = {}
        all_loaded = True
        
        for patch_size in patch_sizes:
            print(f"  Loading patch{patch_size}...")
            model = load_single_model(
                checkpoint_dirs[patch_size], dataset, model_name, patch_size, 
                img_size, num_classes, run_id, device
            )
            
            if model is None:
                all_loaded = False
                break
            else:
                models_dict[str(patch_size)] = model
        
        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for run {run_id}")
            # Clear any loaded models
            for model in models_dict.values():
                del model
            torch.cuda.empty_cache()
            continue
        
        # Early Fusion
        print("  - Early Fusion")
        early_model = EarlyFusionModel(models_dict, num_classes, fusion_dim=256)  # Reduced dimension
        early_model.to(device)
        early_model.eval()  # Set to evaluation mode
        
        # Free up memory from individual models (they are now part of early_model)
        for model in models_dict.values():
            del model
        torch.cuda.empty_cache()
        
        acc, bal_acc, auc = evaluate_fusion_model(
            early_model, test_loader, num_classes, device
        )
        
        all_results.append({
            "dataset": dataset,
            "fusion_method": "early_fusion",
            "patch_size": "-".join(map(str, patch_sizes)),
            "run_id": run_id,
            "model_name": model_name,
            "test_acc": round(acc*100, 2),
            "test_bal_acc": round(bal_acc*100, 2),
            "test_auc": round(auc*100, 2)
        })
        
        print(f"    Early Fusion: Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, AUC={auc*100:.2f}%")
        
        # Delete early model and clear cache
        del early_model
        torch.cuda.empty_cache()
        
        # Reload models for Cross-Attention
        models_dict = {}
        all_loaded = True
        
        for patch_size in patch_sizes:
            print(f"  Loading patch{patch_size} for Cross-Attention...")
            model = load_single_model(
                checkpoint_dirs[patch_size], dataset, model_name, patch_size, 
                img_size, num_classes, run_id, device
            )
            
            if model is None:
                all_loaded = False
                break
            else:
                models_dict[str(patch_size)] = model
        
        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for Cross-Attention run {run_id}")
            for model in models_dict.values():
                del model
            torch.cuda.empty_cache()
            continue
        
        # Cross-Attention Fusion
        print("  - Cross-Attention Fusion")
        cross_model = CrossAttentionFusion(
            models_dict, num_classes, feature_dim=256, num_heads=4  # Reduced dimensions
        )
        cross_model.to(device)
        cross_model.eval()
        
        # Free up memory
        for model in models_dict.values():
            del model
        torch.cuda.empty_cache()
        
        acc, bal_acc, auc = evaluate_fusion_model(
            cross_model, test_loader, num_classes, device
        )
        
        all_results.append({
            "dataset": dataset,
            "fusion_method": "cross_attention",
            "patch_size": "-".join(map(str, patch_sizes)),
            "run_id": run_id,
            "model_name": model_name,
            "test_acc": round(acc*100, 2),
            "test_bal_acc": round(bal_acc*100, 2),
            "test_auc": round(auc*100, 2)
        })
        
        print(f"    Cross-Attention: Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, AUC={auc*100:.2f}%")
        
        # Delete cross model and clear cache
        del cross_model
        torch.cuda.empty_cache()
    
    # 3. Cross-run fusion (different combinations) - Limited to reduce memory usage
    print("\n" + "="*50)
    print("3. Evaluating cross-run fusion (limited combinations)...")
    print("="*50)
    
    # Generate limited combinations to reduce memory usage
    run_combinations = list(product(range(1, robustness + 1), repeat=len(patch_sizes)))
    
    # Limit to fewer combinations
    max_combinations = 5  # Reduced from 10
    if len(run_combinations) > max_combinations:
        print(f"Too many combinations ({len(run_combinations)}), using first {max_combinations}")
        run_combinations = run_combinations[:max_combinations]
    
    for combo_idx, run_combo in enumerate(run_combinations):
        print(f"\nCombination {combo_idx+1}/{len(run_combinations)}: run combo = {run_combo}")
        
        torch.cuda.empty_cache()
        
        # Load models one by one
        models_dict = {}
        all_loaded = True
        
        for patch_idx, patch_size in enumerate(patch_sizes):
            run_id = run_combo[patch_idx]
            print(f"  Loading patch{patch_size} (run{run_id})...")
            
            model = load_single_model(
                checkpoint_dirs[patch_size], dataset, model_name, patch_size, 
                img_size, num_classes, run_id, device
            )
            
            if model is None:
                all_loaded = False
                break
            else:
                models_dict[str(patch_size)] = model
        
        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for combo {run_combo}")
            for model in models_dict.values():
                del model
            torch.cuda.empty_cache()
            continue
        
        # Early Fusion
        print(f"  - Early Fusion (runs: {run_combo})")
        early_model = EarlyFusionModel(models_dict, num_classes, fusion_dim=256)
        early_model.to(device)
        
        for model in models_dict.values():
            del model
        torch.cuda.empty_cache()
        
        acc, bal_acc, auc = evaluate_fusion_model(
            early_model, test_loader, num_classes, device
        )
        
        all_results.append({
            "dataset": dataset,
            "fusion_method": "early_fusion_cross",
            "patch_size": "-".join(map(str, patch_sizes)),
            "run_id": f"{run_combo}",
            "model_name": model_name,
            "test_acc": round(acc*100, 2),
            "test_bal_acc": round(bal_acc*100, 2),
            "test_auc": round(auc*100, 2)
        })
        
        print(f"    Early Fusion: Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, AUC={auc*100:.2f}%")
        
        del early_model
        torch.cuda.empty_cache()
        
        # Reload for Cross-Attention
        models_dict = {}
        all_loaded = True
        
        for patch_idx, patch_size in enumerate(patch_sizes):
            run_id = run_combo[patch_idx]
            print(f"  Loading patch{patch_size} (run{run_id}) for Cross-Attention...")
            
            model = load_single_model(
                checkpoint_dirs[patch_size], dataset, model_name, patch_size, 
                img_size, num_classes, run_id, device
            )
            
            if model is None:
                all_loaded = False
                break
            else:
                models_dict[str(patch_size)] = model
        
        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for Cross-Attention combo {run_combo}")
            for model in models_dict.values():
                del model
            torch.cuda.empty_cache()
            continue
        
        # Cross-Attention Fusion
        print(f"  - Cross-Attention Fusion (runs: {run_combo})")
        cross_model = CrossAttentionFusion(
            models_dict, num_classes, feature_dim=256, num_heads=4
        )
        cross_model.to(device)
        
        for model in models_dict.values():
            del model
        torch.cuda.empty_cache()
        
        acc, bal_acc, auc = evaluate_fusion_model(
            cross_model, test_loader, num_classes, device
        )
        
        all_results.append({
            "dataset": dataset,
            "fusion_method": "cross_attention_cross",
            "patch_size": "-".join(map(str, patch_sizes)),
            "run_id": f"{run_combo}",
            "model_name": model_name,
            "test_acc": round(acc*100, 2),
            "test_bal_acc": round(bal_acc*100, 2),
            "test_auc": round(auc*100, 2)
        })
        
        print(f"    Cross-Attention: Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, AUC={auc*100:.2f}%")
        
        del cross_model
        torch.cuda.empty_cache()
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(fusion_save_dir, f"fusion_results_all_runs_{dataset}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n✓ All results saved to: {results_csv_path}")
    
    # Compute summary statistics (only if we have results)
    if len(results_df) > 0:
        print("\n" + "="*60)
        print("4. Summary of Fusion Results (All Runs)")
        print("="*60)
        
        # Group by fusion method and compute mean/std
        summary_df = results_df.groupby('fusion_method').agg({
            'test_acc': ['mean', 'std'],
            'test_bal_acc': ['mean', 'std'],
            'test_auc': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df.reset_index(inplace=True)
        
        # Format as percentages with ± for std
        for col in ['test_acc_mean', 'test_bal_acc_mean', 'test_auc_mean']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].map(lambda x: f"{x:.2f}%")
        
        for col in ['test_acc_std', 'test_bal_acc_std', 'test_auc_std']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].map(lambda x: f"±{x:.2f}")
        
        summary_df.to_csv(os.path.join(fusion_save_dir, f"fusion_summary_all_runs_{dataset}.csv"), index=False)
        
        print("\nSummary Table:")
        print(summary_df.to_string())
        print("\n")
        
        # Also save detailed per-run results as pivot table
        try:
            detailed_results = results_df.pivot_table(
                index=['run_id', 'patch_size'],
                columns='fusion_method',
                values=['test_acc', 'test_bal_acc', 'test_auc']
            )
            detailed_results.to_csv(os.path.join(fusion_save_dir, f"fusion_detailed_all_runs_{dataset}.csv"))
            print(f"✓ Detailed results saved to: {fusion_save_dir}/fusion_detailed_all_runs_{dataset}.csv")
        except Exception as e:
            print(f"Note: Could not create pivot table: {e}")
    else:
        print("\n⚠️ No results to summarize!")
        summary_df = None
    
    print("\n" + "="*60)
    print("✅ All fusion experiments finished!")
    print("="*60)
    
    return results_df, summary_df
