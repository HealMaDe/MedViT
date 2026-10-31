# fusion_evaluate_3d.py
import os
import glob
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from model_3d import get_model_3d


def find_checkpoint_dirs_3d(base_dir, dataset, model_name, patch_sizes):
   
    checkpoint_dirs = {}
    model_short = "vit_tiny" if "vit_tiny" in model_name else ("vit_small" if "vit_small" in model_name else "vit_base")
    if 'flexivit' in model_name:
        model_short = 'flexivit_base'
    elif 'swin' in model_name:
        model_short = 'swin_tiny'

    dataset_variants = [dataset, f"{dataset}_28"]

    for patch_size in patch_sizes:
        found = False
        for dataset_var in dataset_variants:
            folder_name = f"{dataset_var}_{model_short}_p{patch_size}"
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                checkpoint_dirs[patch_size] = folder_path
                print(f"  ✓ Found checkpoint directory for patch{patch_size}: {folder_path}")
                found = True
                break
        if not found:
            print(f"  ✗ Warning: Directory not found for patch{patch_size}")
            search_pattern = os.path.join(base_dir, f"*_{model_short}_p{patch_size}")
            matching_dirs = glob.glob(search_pattern)
            if matching_dirs:
                print(f"    Did you mean: {', '.join(matching_dirs)}")
    return checkpoint_dirs


def load_single_model_3d(checkpoint_dir, dataset, model_name, patch_size,
                          img_size, num_classes, run_id, device):
    
    folder_path = checkpoint_dir
    tag = model_name.replace("/", "_").replace(".", "_")

    patterns = [
        f"best_{dataset}_repeat_patch{patch_size}_run{run_id}.pth",
        f"best_{dataset}_{tag}_dynpatch_run{run_id}.pth", 
        f"*{tag}*patch{patch_size}*run{run_id}*.pth",       
    ]

    checkpoint_path = None
    for pattern in patterns:
        matches = glob.glob(os.path.join(folder_path, pattern))
        if matches:
            checkpoint_path = matches[0]
            print(f"  Found: {os.path.basename(checkpoint_path)}")
            break

    if checkpoint_path is None:
        print(f"  ✗ Checkpoint not found for patch{patch_size}, run{run_id} in {folder_path}")
        return None

    try:
        model, _ = get_model_3d(
            model_name, num_classes, patch_size, img_size, "cpu", pretrained=False
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"  ✓ Successfully loaded patch{patch_size}, run{run_id}")
        return model
    except Exception as e:
        print(f"  ✗ Error loading checkpoint {checkpoint_path}: {e}")
        return None


def evaluate_single_model_3d(model, test_loader, num_classes, device):
    model.eval()
    all_labels, all_preds_class, all_probs = [], [], []

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

    acc = np.mean(np.array(all_labels) == np.array(all_preds_class))
    bal_acc = balanced_accuracy_score(all_labels, all_preds_class)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs))
        else:
            all_probs_array = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs_array, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"AUC Error: {e}")
        auc = 0.0

    return acc, bal_acc, auc


def evaluate_fusion_model_3d(model, test_loader, num_classes, device):
    model.eval()
    all_labels, all_preds_class, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds_class = probs.argmax(dim=1)

            probs_np = probs.cpu().numpy()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds_class.extend(preds_class.cpu().numpy().tolist())

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


def evaluate_late_fusion_model_3d(model, test_loader, num_classes, device):
    model.eval()
    all_labels, all_preds_class, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            probs = model(imgs, return_log_probs=False)
            preds_class = probs.argmax(dim=1)

            probs_np = probs.cpu().numpy()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds_class.extend(preds_class.cpu().numpy().tolist())

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
