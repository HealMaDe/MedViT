import os, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt

from src.data_loader import MedMNISTDataset, get_transforms
from src.model import get_model
from src.utils import save_logs, plot_curves


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.long().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_time = time.time() - t0
    return running_loss / total, correct / total, epoch_time


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds_class, all_probs = [], [], []

    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.long().to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds_class = outputs.argmax(1)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds_class.extend(preds_class.cpu().numpy().tolist())
        if probs.shape[1] == 2:
            all_probs.extend(probs[:, 1].tolist())
        else:
            all_probs.extend(probs.tolist())

    elapsed = time.time() - t0
    fps = len(all_labels) / elapsed if elapsed > 0 else 0.0

    acc = np.mean(np.array(all_labels) == np.array(all_preds_class))
    bal_acc = balanced_accuracy_score(all_labels, all_preds_class)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs))
        else:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, bal_acc, auc, elapsed, fps


def run_experiment(cfg, device):
    dataset = cfg["dataset"]
    img_size = cfg["img_size"]
    patch_sizes = cfg["patch_sizes"]
    models = cfg["models"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    scheduler_step = cfg["scheduler_step"]
    scheduler_gamma = cfg["scheduler_gamma"]
    robustness = cfg.get("robustness", 1)
    save_dir = cfg["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # Dataset
    npz_path = f"./data/{dataset}.npz"
    train_tf, val_tf = get_transforms(img_size)
    train_ds = MedMNISTDataset("train", dataset, npz_path, transform=train_tf)
    val_ds   = MedMNISTDataset("val", dataset, npz_path, transform=val_tf)
    test_ds  = MedMNISTDataset("test", dataset, npz_path, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(set(train_ds.labels.flatten()))
    num_test_samples = len(test_ds)

    # Robustness loop
    all_runs = []
    for run_id in range(1, robustness + 1):
        print(f"\n=== Run {run_id}/{robustness} for {dataset} ===")

        for model_name in models:
            for patch_size in patch_sizes:
                print(f"\n>>> Training {model_name}, patch {patch_size}, img {img_size}")

                model = get_model(model_name, num_classes, patch_size, img_size, device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

                history = {"train_loss": [], "train_acc": [],
                           "val_loss": [], "val_acc": [], "val_auc": []}

                best_val_loss = float("inf")
                best_ckpt = None

                total_train_time = 0.0
                for epoch in range(1, epochs + 1):
                    train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
                    total_train_time += epoch_time
                    val_loss, val_acc, val_bal_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, num_classes, device)

                    print(f"Epoch {epoch}/{epochs} | LR {optimizer.param_groups[0]['lr']:.2e} "
                          f"| Train loss {train_loss:.4f} / Train acc {train_acc*100:.2f}% "
                          f"| Val loss {val_loss:.4f} / Val acc {val_acc*100:.2f}% / Val bal_acc {val_bal_acc*100:.2f}% / val auc {val_auc*100:.2f}%")

                    history["train_loss"].append(train_loss)
                    history["train_acc"].append(round(train_acc*100,2))
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(round(val_acc*100,2))
                    history["val_auc"].append(round(val_auc*100,2))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt = f"{save_dir}/best_{dataset}_{model_name}_p{patch_size}_run{run_id}.pth"
                        torch.save(model.state_dict(), best_ckpt)

                    scheduler.step()

                # Load best model and test
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                test_loss, test_acc, test_bal_acc, test_auc, test_time, fps = evaluate(
                    model, test_loader, criterion, num_classes, device
                )
                
                test_time_per_img = (test_time / num_test_samples) * 1000   # ms
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else "N/A"

                run_result = {
                    "dataset": dataset,
                    "model": model_name,
                    "patch_size": patch_size,
                    "run_id": run_id,
                    "test_loss": test_loss,
                    "test_acc": round(test_acc*100,2),
                    "test_bal_acc": round(test_bal_acc*100,2),
                    "test_auc": round(test_auc*100,2),
                    "train_time": round(total_train_time,2),
                    "test_time_per_image": round(test_time_per_img,2),
                    "fps": round(fps,2),
                    "vram_mb": peak_vram_mb,
                }
                all_runs.append(run_result)

                # Save logs & plots only for best checkpoint
                log_path = f"{save_dir}/log_{dataset}_{model_name}_p{patch_size}_run{run_id}.csv"
                plot_path = f"{save_dir}/curves_{dataset}_{model_name}_p{patch_size}_run{run_id}.png"
                save_logs(history, log_path)
                plot_curves(history, plot_path)

    # --- Aggregate robustness results ---
    if robustness > 1:
        import pandas as pd
        df = pd.DataFrame(all_runs)
        agg_df = df.groupby(["dataset", "model", "patch_size"]).agg(["mean", "std"]).reset_index()
        agg_df.columns = ["_".join(col).rstrip("_") for col in agg_df.columns.values]
        
        avg_path = f"{save_dir}/avg_results_{dataset}.csv"
        agg_df.to_csv(avg_path, index=False)
        print(f"\nSaved averaged robustness results to {avg_path}")

    return all_runs





