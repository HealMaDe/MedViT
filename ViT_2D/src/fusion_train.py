

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MedMNISTDataset, get_transforms
from model import get_model
from fusion_model import EarlyFusionModel, CrossAttentionFusion, LateFusionModel, get_trainable_parameters
from fusion_evaluate import load_single_model, evaluate_fusion_model, find_checkpoint_dirs
import numpy as np
import pandas as pd



def train_fusion_model(model, train_loader, val_loader, num_classes, device,
                       epochs=20, lr=1e-3, save_path=None, model_name="fusion",
                       weight_decay=0.05):
    
    trainable_params = get_trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"پارامترهای trainable: {n_trainable:,} / کل: {n_total:,} "
          f"({n_trainable/n_total*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Model: {model_name}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()  
        train_loss = 0
        train_correct = 0
        train_total = 0

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
        val_loss = 0
        val_correct = 0
        val_total = 0

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
        print(f"\n✅ Best model from epoch {best_epoch}:")
        print(f"   Validation Loss: {best_val_loss:.4f} (LOWEST)")
        print(f"   Validation Accuracy: {best_val_acc:.2f}%")

        if save_path:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'model_name': model_name
            }, save_path)
            print(f"✅ Best model saved to: {save_path}")

    return model, best_val_loss, best_val_acc


 
def train_late_fusion_model(model, train_loader, val_loader, num_classes, device,
                             epochs=30, lr=0.05, save_path=None, model_name="late_fusion",
                             weight_decay=0.0):

    trainable_params = get_trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"پارامترهای trainable: {n_trainable} (فقط وزن هر branch)")
 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0
 
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Model: {model_name}")
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
 
        weights_str = ", ".join(f"{w:.3f}" for w in model.get_branch_weights().tolist()) \
            if model.weighting == "scalar" else "(per_class)"
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
        print(f"\n✅ Best model from epoch {best_epoch}:")
        print(f"   Validation Loss: {best_val_loss:.4f} (LOWEST)")
        print(f"   Validation Accuracy: {best_val_acc:.2f}%")
        print(f"   Final branch weights: {model.get_branch_weights().tolist()}")
 
        if save_path:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'model_name': model_name
            }, save_path)
            print(f"✅ Best model saved to: {save_path}")
 
    return model, best_val_loss, best_val_acc
 
 
def evaluate_late_fusion_model(model, test_loader, num_classes, device):
    
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
 
    model.eval()
    all_labels, all_preds_class, all_probs = [], [], []
 
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
 
            probs = model(imgs, return_log_probs=False)  # probability مستقیم
            preds_class = probs.argmax(dim=1)
 
            probs_np = probs.cpu().numpy()
            preds_np = preds_class.cpu().numpy()
            labels_np = labels.cpu().numpy()
 
            all_labels.extend(labels_np.tolist())
            all_preds_class.extend(preds_np.tolist())
 
            if num_classes == 2:
                all_probs.extend(probs_np[:, 1].tolist())
            else:
                all_probs.extend(probs_np.tolist())
 
    all_labels = np.array(all_labels)
    all_preds_class = np.array(all_preds_class)
 
    acc = np.mean(all_labels == all_preds_class)
    bal_acc = balanced_accuracy_score(all_labels, all_preds_class)
 
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs))
        else:
            all_probs = np.array(all_probs)
            row_sums = all_probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-5):
                all_probs = all_probs / row_sums.reshape(-1, 1)
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"  AUC Error: {e}")
        auc = 0.0
 
    return acc, bal_acc, auc
 



def run_fusion_training(cfg, device):
    
    dataset = cfg["dataset"]
    img_size = cfg["img_size"]
    patch_sizes = cfg["patch_sizes"]
    models = cfg["models"]
    batch_size = cfg.get("batch_size", 16)
    save_dir = cfg["save_dir"]
    checkpoint_base_dir = cfg.get("checkpoint_dir", save_dir)
    robustness = cfg.get("robustness", 3)

    fusion_save_dir = os.path.join(save_dir, "fusion_results")
    os.makedirs(fusion_save_dir, exist_ok=True)

    npz_path = f"/mnt/18T/healmade/vit_patch/MedViT/ViT_2D/data/{dataset}.npz"
    train_tf, val_tf = get_transforms(img_size)

    train_ds = MedMNISTDataset("train", dataset, npz_path, transform=train_tf)
    val_ds = MedMNISTDataset("val", dataset, npz_path, transform=val_tf)
    test_ds = MedMNISTDataset("test", dataset, npz_path, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(set(train_ds.labels.flatten()))
    model_name = models[0]

    print("\n" + "="*80)
    print("FINDING CHECKPOINT DIRECTORIES")
    print("="*80)
    checkpoint_dirs = find_checkpoint_dirs(
        checkpoint_base_dir, dataset, model_name, patch_sizes
    )

    if not checkpoint_dirs:
        print("❌ No checkpoint directories found!")
        return None, None

    early_results = []
    cross_results = []
    late_results = []

    print("\n" + "="*80)
    print("TRAINING FUSION MODELS")
    print("Selection criteria: LOWEST VALIDATION LOSS")
    print("="*80)

    for run_id in range(1, robustness + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_id}/{robustness}")
        print(f"{'='*80}")

        # ============================================
        # 1. Train Early Fusion
        # ============================================
        print("\n" + "-"*80)
        print("EARLY FUSION")
        print("-"*80)

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
            models_dict[str(patch_size)] = model

        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for run {run_id}")
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()
            continue

        early_model = EarlyFusionModel(models_dict, num_classes, fusion_dim=256)
        early_model.to(device)

        for m in models_dict.values():
            del m
        torch.cuda.empty_cache()

        early_save_path = os.path.join(fusion_save_dir, f"early_fusion_run{run_id}_best.pth")
        early_model, best_val_loss, best_val_acc = train_fusion_model(
            early_model, train_loader, val_loader, num_classes, device,
            epochs=20, lr=1e-3, save_path=early_save_path,
            model_name=f"EarlyFusion_Run{run_id}"
        )

        print("\n  📊 Evaluating on Test Set...")
        test_acc, test_bal_acc, test_auc = evaluate_fusion_model(
            early_model, test_loader, num_classes, device
        )

        early_results.append({
            "run_id": run_id,
            "test_acc": test_acc * 100,
            "test_bal_acc": test_bal_acc * 100,
            "test_auc": test_auc * 100,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc
        })

        print(f"\n  ✅ Early Fusion Results (Run {run_id}):")
        print(f"     Test Accuracy:   {test_acc*100:.2f}%")
        print(f"     Test Bal Acc:    {test_bal_acc*100:.2f}%")
        print(f"     Test AUC:        {test_auc*100:.2f}%")

        del early_model
        torch.cuda.empty_cache()

        # ============================================
        # 2. Train Cross-Attention Fusion
        # ============================================
        print("\n" + "-"*80)
        print("CROSS-ATTENTION FUSION")
        print("-"*80)

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
            models_dict[str(patch_size)] = model

        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for Cross-Attention run {run_id}")
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()
            continue

        cross_model = CrossAttentionFusion(
            models_dict, num_classes, feature_dim=256, num_heads=4
        )
        cross_model.to(device)

        for m in models_dict.values():
            del m
        torch.cuda.empty_cache()

        cross_save_path = os.path.join(fusion_save_dir, f"cross_attention_run{run_id}_best.pth")
        cross_model, best_val_loss, best_val_acc = train_fusion_model(
            cross_model, train_loader, val_loader, num_classes, device,
            epochs=30, lr=5e-4,
            weight_decay=0.05,
            save_path=cross_save_path,
            model_name=f"CrossAttention_Run{run_id}"
        )

        print("\n  📊 Evaluating on Test Set...")
        test_acc, test_bal_acc, test_auc = evaluate_fusion_model(
            cross_model, test_loader, num_classes, device
        )

        cross_results.append({
            "run_id": run_id,
            "test_acc": test_acc * 100,
            "test_bal_acc": test_bal_acc * 100,
            "test_auc": test_auc * 100,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc
        })

        print(f"\n  ✅ Cross-Attention Results (Run {run_id}):")
        print(f"     Test Accuracy:   {test_acc*100:.2f}%")
        print(f"     Test Bal Acc:    {test_bal_acc*100:.2f}%")
        print(f"     Test AUC:        {test_auc*100:.2f}%")

        del cross_model
        torch.cuda.empty_cache()


        # ============================================
        # 3. Train Late Fusion
        # ============================================
        print("\n" + "-"*80)
        print("LATE FUSION")
        print("-"*80)
        
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
            models_dict[str(patch_size)] = model
        
        if not all_loaded or len(models_dict) != len(patch_sizes):
            print(f"  Warning: Could not load all models for run {run_id}")
            for m in models_dict.values():
                del m
            torch.cuda.empty_cache()
            continue
        
        late_model = LateFusionModel(models_dict, num_classes, weighting="scalar")
        late_model.to(device)
        
        for m in models_dict.values():
            del m
        torch.cuda.empty_cache()
        
        late_save_path = os.path.join(fusion_save_dir, f"late_fusion_run{run_id}_best.pth")
        late_model, best_val_loss, best_val_acc = train_late_fusion_model(
            late_model, train_loader, val_loader, num_classes, device,
            epochs=20, lr=0.05, save_path=late_save_path,
            model_name=f"LateFusion_Run{run_id}"
        )
        
        print("\n  📊 Evaluating on Test Set...")
        test_acc, test_bal_acc, test_auc = evaluate_late_fusion_model(
            late_model, test_loader, num_classes, device
        )
        
        late_results.append({
            "run_id": run_id,
            "test_acc": test_acc * 100,
            "test_bal_acc": test_bal_acc * 100,
            "test_auc": test_auc * 100,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc
        })
        
        print(f"\n  ✅ Late Fusion Results (Run {run_id}):")
        print(f"     Test Accuracy:   {test_acc*100:.2f}%")
        print(f"     Test Bal Acc:    {test_bal_acc*100:.2f}%")
        print(f"     Test AUC:        {test_auc*100:.2f}%")
        
        del late_model
        torch.cuda.empty_cache()




    # ============================================
    # AGGREGATE RESULTS ACROSS ALL RUNS
    # ============================================
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (Across All Runs)")
    print("="*80)

    aggregated_results = []

    if early_results:
        early_df = pd.DataFrame(early_results)
        aggregated_results.append({
            "dataset": dataset,
            "fusion_method": "early_fusion",
            "patch_size": "-".join(map(str, patch_sizes)),
            "test_acc_mean": round(early_df['test_acc'].mean(), 2),
            "test_acc_std": round(early_df['test_acc'].std(), 2),
            "test_bal_acc_mean": round(early_df['test_bal_acc'].mean(), 2),
            "test_bal_acc_std": round(early_df['test_bal_acc'].std(), 2),
            "test_auc_mean": round(early_df['test_auc'].mean(), 2),
            "test_auc_std": round(early_df['test_auc'].std(), 2),
        })
        print(f"\n📊 EARLY FUSION (over {len(early_results)} runs):")
        print(f"   Test Acc:     {early_df['test_acc'].mean():.2f}% ± {early_df['test_acc'].std():.2f}")
        print(f"   Test Bal Acc: {early_df['test_bal_acc'].mean():.2f}% ± {early_df['test_bal_acc'].std():.2f}")
        print(f"   Test AUC:     {early_df['test_auc'].mean():.2f}% ± {early_df['test_auc'].std():.2f}")

    if cross_results:
        cross_df = pd.DataFrame(cross_results)
        aggregated_results.append({
            "dataset": dataset,
            "fusion_method": "cross_attention",
            "patch_size": "-".join(map(str, patch_sizes)),
            "test_acc_mean": round(cross_df['test_acc'].mean(), 2),
            "test_acc_std": round(cross_df['test_acc'].std(), 2),
            "test_bal_acc_mean": round(cross_df['test_bal_acc'].mean(), 2),
            "test_bal_acc_std": round(cross_df['test_bal_acc'].std(), 2),
            "test_auc_mean": round(cross_df['test_auc'].mean(), 2),
            "test_auc_std": round(cross_df['test_auc'].std(), 2),
        })
        print(f"\n📊 CROSS-ATTENTION (over {len(cross_results)} runs):")
        print(f"   Test Acc:     {cross_df['test_acc'].mean():.2f}% ± {cross_df['test_acc'].std():.2f}")
        print(f"   Test Bal Acc: {cross_df['test_bal_acc'].mean():.2f}% ± {cross_df['test_bal_acc'].std():.2f}")
        print(f"   Test AUC:     {cross_df['test_auc'].mean():.2f}% ± {cross_df['test_auc'].std():.2f}")

    if late_results:
        late_df = pd.DataFrame(late_results)
        aggregated_results.append({
            "dataset": dataset,
            "fusion_method": "late_fusion",
            "patch_size": "-".join(map(str, patch_sizes)),
            "test_acc_mean": round(late_df['test_acc'].mean(), 2),
            "test_acc_std": round(late_df['test_acc'].std(), 2),
            "test_bal_acc_mean": round(late_df['test_bal_acc'].mean(), 2),
            "test_bal_acc_std": round(late_df['test_bal_acc'].std(), 2),
            "test_auc_mean": round(late_df['test_auc'].mean(), 2),
            "test_auc_std": round(late_df['test_auc'].std(), 2),
        })
        print(f"\n📊 LATE FUSION (over {len(late_results)} runs):")
        print(f"   Test Acc:     {late_df['test_acc'].mean():.2f}% ± {late_df['test_acc'].std():.2f}")
        print(f"   Test Bal Acc: {late_df['test_bal_acc'].mean():.2f}% ± {late_df['test_bal_acc'].std():.2f}")
        print(f"   Test AUC:     {late_df['test_auc'].mean():.2f}% ± {late_df['test_auc'].std():.2f}")



    if not aggregated_results:
        return None, None

    agg_df = pd.DataFrame(aggregated_results)
    agg_csv_path = os.path.join(fusion_save_dir, f"fusion_aggregated_results_{dataset}.csv")
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"\n✅ Aggregated results saved to: {agg_csv_path}")

    all_runs = []
    for r in early_results:
        r = {**r, 'fusion_method': 'early_fusion'}
        all_runs.append(r)
    for r in cross_results:
        r = {**r, 'fusion_method': 'cross_attention'}
        all_runs.append(r)

    for r in late_results:
        r = {**r, 'fusion_method': 'late_fusion'}
        all_runs.append(r)

    runs_df = pd.DataFrame(all_runs)
    runs_df = runs_df[['fusion_method', 'run_id', 'test_acc', 'test_bal_acc', 'test_auc', 'best_val_loss', 'best_val_acc']]
    runs_csv_path = os.path.join(fusion_save_dir, f"fusion_per_run_results_{dataset}.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"✅ Per-run results saved to: {runs_csv_path}")

    paper_df = agg_df[['dataset', 'fusion_method', 'patch_size',
                       'test_acc_mean', 'test_acc_std',
                       'test_bal_acc_mean', 'test_bal_acc_std',
                       'test_auc_mean', 'test_auc_std']].copy()

    paper_df['test_acc'] = paper_df['test_acc_mean'].astype(str) + '% ± ' + paper_df['test_acc_std'].astype(str)
    paper_df['test_bal_acc'] = paper_df['test_bal_acc_mean'].astype(str) + '% ± ' + paper_df['test_bal_acc_std'].astype(str)
    paper_df['test_auc'] = paper_df['test_auc_mean'].astype(str) + '% ± ' + paper_df['test_auc_std'].astype(str)

    paper_df = paper_df[['dataset', 'fusion_method', 'patch_size', 'test_acc', 'test_bal_acc', 'test_auc']]
    paper_csv_path = os.path.join(fusion_save_dir, f"fusion_paper_table_{dataset}.csv")
    paper_df.to_csv(paper_csv_path, index=False)
    print(f"✅ Paper-ready table saved to: {paper_csv_path}")

    print("\n" + "="*80)
    print("FINAL AGGREGATED RESULTS")
    print("="*80)
    print(agg_df.to_string(index=False))

    print("\n" + "="*80)
    print("PAPER-READY TABLE")
    print("="*80)
    print(paper_df.to_string(index=False))

    return agg_df, runs_df
