
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader_3d import get_loaders

from fusion_model_3d import EarlyFusionModel, CrossAttentionFusion, LateFusionModel, get_trainable_parameters
from fusion_evaluate_3d import (
    find_checkpoint_dirs_3d, load_single_model_3d,
    evaluate_fusion_model_3d, evaluate_late_fusion_model_3d,
)

import numpy as np
import pandas as pd


def train_fusion_model_3d(model, train_loader, val_loader, num_classes, device,
                           epochs=20, lr=1e-3, save_path=None, model_name="fusion",
                           weight_decay=0.05):
    
    trainable_params = get_trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f" trainable: {n_trainable:,} / total: {n_total:,} "
          f"({n_trainable/n_total*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    print(f"\nStarting training for {epochs} epochs...  Model: {model_name}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.long().to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / val_total
        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            print(f"  ✓ New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Best model from epoch {best_epoch}: Val Loss={best_val_loss:.4f}, Val Acc={best_val_acc:.2f}%")
        if save_path:
            torch.save({'epoch': best_epoch, 'model_state_dict': best_model_state,
                        'val_loss': best_val_loss, 'val_acc': best_val_acc,
                        'model_name': model_name}, save_path)
            print(f"✅ Best model saved to: {save_path}")

    return model, best_val_loss, best_val_acc


def train_late_fusion_model_3d(model, train_loader, val_loader, num_classes, device,
                                epochs=30, lr=0.05, save_path=None, model_name="late_fusion",
                                weight_decay=0.0):
    
    trainable_params = get_trainable_parameters(model)
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    print(f"\nStarting training for {epochs} epochs...  Model: {model_name}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            log_probs = model(imgs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = log_probs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.long().to(device)
                log_probs = model(imgs)
                loss = criterion(log_probs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, predicted = log_probs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / val_total
        scheduler.step()

        weights_str = ", ".join(f"{w:.3f}" for w in model.get_branch_weights().tolist())
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%  | branch_weights=[{weights_str}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            print(f"  ✓ New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Best model from epoch {best_epoch}: Val Loss={best_val_loss:.4f}, Val Acc={best_val_acc:.2f}%")
        print(f"   Final branch weights: {model.get_branch_weights().tolist()}")
        if save_path:
            torch.save({'epoch': best_epoch, 'model_state_dict': best_model_state,
                        'val_loss': best_val_loss, 'val_acc': best_val_acc,
                        'model_name': model_name}, save_path)
            print(f"✅ Best model saved to: {save_path}")

    return model, best_val_loss, best_val_acc


def run_fusion_training_3d(cfg, device):
    dataset = cfg["dataset"]
    img_size = cfg["img_size"]
    patch_sizes = cfg["patch_sizes"]
    model_name = cfg["model_name"]
    batch_size = cfg.get("batch_size", 8)
    save_dir = cfg["save_dir"]
    checkpoint_base_dir = cfg.get("checkpoint_dir", save_dir)
    robustness = cfg.get("robustness", 3)

    fusion_save_dir = os.path.join(save_dir, "fusion_results")
    os.makedirs(fusion_save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = get_loaders(dataset, batch_size)

    print("\n" + "="*80)
    print("FINDING CHECKPOINT DIRECTORIES (3D)")
    print("="*80)
    checkpoint_dirs = find_checkpoint_dirs_3d(checkpoint_base_dir, dataset, model_name, patch_sizes)

    if not checkpoint_dirs:
        print("❌ No checkpoint directories found!")
        return None, None

    early_results, cross_results, late_results = [], [], []

    for run_id in range(1, robustness + 1):
        print(f"\n{'='*80}\nRUN {run_id}/{robustness}\n{'='*80}")

        # ---------------- Early Fusion ----------------
        print("\n" + "-"*80 + "\nEARLY FUSION (3D)\n" + "-"*80)
        models_dict = {}
        all_loaded = True
        for patch_size in patch_sizes:
            model = load_single_model_3d(checkpoint_dirs[patch_size], dataset, model_name,
                                          patch_size, img_size, num_classes, run_id, device)
            if model is None:
                all_loaded = False
                break
            models_dict[str(patch_size)] = model

        if all_loaded and len(models_dict) == len(patch_sizes):
            early_model = EarlyFusionModel(models_dict, num_classes, fusion_dim=256).to(device)
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()

            early_save_path = os.path.join(fusion_save_dir, f"early_fusion_run{run_id}_best.pth")
            early_model, bvl, bva = train_fusion_model_3d(
                early_model, train_loader, val_loader, num_classes, device,
                epochs=20, lr=1e-3, save_path=early_save_path,
                model_name=f"EarlyFusion3D_Run{run_id}"
            )
            test_acc, test_bal_acc, test_auc = evaluate_fusion_model_3d(early_model, test_loader, num_classes, device)
            early_results.append({"run_id": run_id, "test_acc": test_acc*100,
                                   "test_bal_acc": test_bal_acc*100, "test_auc": test_auc*100,
                                   "best_val_loss": bvl, "best_val_acc": bva})
            print(f"\n  ✅ Early Fusion (Run {run_id}): Acc={test_acc*100:.2f}% BalAcc={test_bal_acc*100:.2f}% AUC={test_auc*100:.2f}%")
            del early_model
            torch.cuda.empty_cache()
        else:
            print(f"  Warning: Could not load all models for Early Fusion run {run_id}")

        # ---------------- Cross-Attention Fusion ----------------
        print("\n" + "-"*80 + "\nCROSS-ATTENTION FUSION (3D)\n" + "-"*80)
        models_dict = {}
        all_loaded = True
        for patch_size in patch_sizes:
            model = load_single_model_3d(checkpoint_dirs[patch_size], dataset, model_name,
                                          patch_size, img_size, num_classes, run_id, device)
            if model is None:
                all_loaded = False
                break
            models_dict[str(patch_size)] = model

        if all_loaded and len(models_dict) == len(patch_sizes):
            cross_model = CrossAttentionFusion(models_dict, num_classes, feature_dim=256, num_heads=4).to(device)
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()

            cross_save_path = os.path.join(fusion_save_dir, f"cross_attention_run{run_id}_best.pth")
            cross_model, bvl, bva = train_fusion_model_3d(
                cross_model, train_loader, val_loader, num_classes, device,
                epochs=40, lr=5e-4, save_path=cross_save_path,
                model_name=f"CrossAttention3D_Run{run_id}"
            )
            test_acc, test_bal_acc, test_auc = evaluate_fusion_model_3d(cross_model, test_loader, num_classes, device)
            cross_results.append({"run_id": run_id, "test_acc": test_acc*100,
                                   "test_bal_acc": test_bal_acc*100, "test_auc": test_auc*100,
                                   "best_val_loss": bvl, "best_val_acc": bva})
            print(f"\n  ✅ Cross-Attention (Run {run_id}): Acc={test_acc*100:.2f}% BalAcc={test_bal_acc*100:.2f}% AUC={test_auc*100:.2f}%")
            del cross_model
            torch.cuda.empty_cache()
        else:
            print(f"  Warning: Could not load all models for Cross-Attention run {run_id}")

        # ---------------- Late Fusion ----------------
        print("\n" + "-"*80 + "\nLATE FUSION (3D)\n" + "-"*80)
        models_dict = {}
        all_loaded = True
        for patch_size in patch_sizes:
            model = load_single_model_3d(checkpoint_dirs[patch_size], dataset, model_name,
                                          patch_size, img_size, num_classes, run_id, device)
            if model is None:
                all_loaded = False
                break
            models_dict[str(patch_size)] = model

        if all_loaded and len(models_dict) == len(patch_sizes):
            late_model = LateFusionModel(models_dict, num_classes, weighting="scalar").to(device)
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()

            late_save_path = os.path.join(fusion_save_dir, f"late_fusion_run{run_id}_best.pth")
            late_model, bvl, bva = train_late_fusion_model_3d(
                late_model, train_loader, val_loader, num_classes, device,
                epochs=30, lr=0.05, save_path=late_save_path,
                model_name=f"LateFusion3D_Run{run_id}"
            )
            test_acc, test_bal_acc, test_auc = evaluate_late_fusion_model_3d(late_model, test_loader, num_classes, device)
            late_results.append({"run_id": run_id, "test_acc": test_acc*100,
                                  "test_bal_acc": test_bal_acc*100, "test_auc": test_auc*100,
                                  "best_val_loss": bvl, "best_val_acc": bva})
            print(f"\n  ✅ Late Fusion (Run {run_id}): Acc={test_acc*100:.2f}% BalAcc={test_bal_acc*100:.2f}% AUC={test_auc*100:.2f}%")
            print(f"     Branch weights: {late_model.get_branch_weights().tolist()}")
            del late_model
            torch.cuda.empty_cache()
        else:
            print(f"  Warning: Could not load all models for Late Fusion run {run_id}")

    # ------------- Aggregate -------------
    aggregated_results = []
    for name, results in [("early_fusion", early_results), ("cross_attention", cross_results), ("late_fusion", late_results)]:
        if results:
            df = pd.DataFrame(results)
            aggregated_results.append({
                "dataset": dataset, "fusion_method": name,
                "patch_size": "-".join(map(str, patch_sizes)),
                "test_acc_mean": round(df['test_acc'].mean(), 2), "test_acc_std": round(df['test_acc'].std(), 2),
                "test_bal_acc_mean": round(df['test_bal_acc'].mean(), 2), "test_bal_acc_std": round(df['test_bal_acc'].std(), 2),
                "test_auc_mean": round(df['test_auc'].mean(), 2), "test_auc_std": round(df['test_auc'].std(), 2),
            })
            print(f"\n📊 {name.upper()} (over {len(results)} runs): "
                  f"Acc={df['test_acc'].mean():.2f}%±{df['test_acc'].std():.2f} | "
                  f"BalAcc={df['test_bal_acc'].mean():.2f}%±{df['test_bal_acc'].std():.2f} | "
                  f"AUC={df['test_auc'].mean():.2f}%±{df['test_auc'].std():.2f}")

    if not aggregated_results:
       
        return None, None

    agg_df = pd.DataFrame(aggregated_results)
    agg_csv_path = os.path.join(fusion_save_dir, f"fusion_aggregated_results_3d_{dataset}.csv")
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"\n✅ Aggregated results saved to: {agg_csv_path}")

    all_runs = []
    for name, results in [("early_fusion", early_results), ("cross_attention", cross_results), ("late_fusion", late_results)]:
        for r in results:
            all_runs.append({**r, 'fusion_method': name})
    runs_df = pd.DataFrame(all_runs)
    runs_csv_path = os.path.join(fusion_save_dir, f"fusion_per_run_results_3d_{dataset}.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"✅ Per-run results saved to: {runs_csv_path}")

    print("\n" + "="*80 + "\nFINAL AGGREGATED RESULTS (3D)\n" + "="*80)
    print(agg_df.to_string(index=False))

    return agg_df, runs_df


if __name__ == "__main__":
    cfg = {
        "dataset": "vesselmnist3d",
        "img_size": 28,
        "patch_sizes": [1, 2, 4],
        "model_name": "vit_tiny_patch16_224",
        "robustness": 3,
        "batch_size": 8,
        "checkpoint_dir": "/mnt/2T/healmade/MedViT/ViT_3D/results",
        "save_dir": "/mnt/2T/healmade/MedViT/ViT_3D/results",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agg_df, runs_df = run_fusion_training_3d(cfg, device)
