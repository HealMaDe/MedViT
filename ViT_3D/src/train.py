import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import os

from src.data_loader_3d import MedMNIST3DDataset, get_loaders
from src.model_3d import ViT3D
from src.utils import save_logs, save_probs, plot_curves, save_extended_summary


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total



@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, return_probs=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(1)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs)

    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
    except:
        auc = 0.0

    if return_probs:
        return (running_loss / len(loader.dataset)), acc, bal_acc, auc, list(zip(all_labels, all_probs))
    return (running_loss / len(loader.dataset)), acc, bal_acc, auc



def run_experiment(cfg, device):
    dataset_name = cfg["dataset"]
    img_size = cfg["img_size"]
    patch_size = cfg["patch_size"]
    inflate_method = cfg["inflate_method"]
    model_size = cfg["model_size"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    scheduler_step = cfg["scheduler_step"]
    scheduler_gamma = cfg["scheduler_gamma"]
    robustness = cfg.get("robustness", 1)
    save_dir = cfg["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # Dataset
    npz_path = f"../data/{dataset_name}.npz"
    
    all_results = []
    all_train_times, all_test_times, all_fps, all_vram = [], [], [], []

    for run in range(robustness):
        print(f"\n=== Run {run+1}/{robustness} for {dataset_name} ===")
        print(f"Patch size: {patch_size}, Inflation: {inflate_method}")

        train_loader, val_loader, test_loader, num_classes = get_loaders(dataset_name,batch_size)

        # Create ViT-3D
        model = ViT3D(
            model_size=model_size,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            pretrained=True,
            inflate_method=inflate_method,
            device=device)

        optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_auc": []}
        best_val_loss, best_ckpt = float("inf"), None

        torch.cuda.reset_peak_memory_stats(device=DEVICE) if DEVICE == "cuda" else None
        train_start = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_bal_acc, val_auc = evaluate(model, val_loader, criterion, num_classes)

            print(f"Epoch {epoch}/{epochs} | LR {optimizer.param_groups[0]['lr']:.2e} "
                  f"| Train loss {train_loss:.2f} / Train acc {train_acc*100:.2f}% "
                  f"| Val loss {val_loss:.2f} / Val acc {val_acc*100:.2f}% "
                  f"/ Val bal_acc {val_bal_acc*100:.2f}% / Val AUC {val_auc*100:.2f}%")

            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc * 100, 2))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc * 100, 2))
            history["val_bal_acc"].append(round(val_bal_acc * 100, 2))
            history["val_auc"].append(round(val_auc * 100, 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = f"{save_dir}/best_{dataset_name}_{inflate_method}_patch{patch_size}_run{run+1}.pth"
                torch.save(model.state_dict(), best_ckpt)

            scheduler.step()

        train_time = time.time() - train_start
        vram_used = torch.cuda.max_memory_allocated(device=DEVICE) / (1024**2) if DEVICE == "cuda" else 0

        # === Test Evaluation with timing + probabilities ===
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        test_start = time.time()
        test_loss, test_acc, test_bal_acc, test_auc, test_probs = evaluate(model, test_loader, criterion, num_classes, return_probs=True)
        test_time = time.time() - test_start

        total_images = len(test_loader.dataset)
        test_time_per_image = test_time / total_images
        fps = total_images / test_time

        all_results.append([test_loss, test_acc, test_bal_acc, test_auc])
        all_train_times.append(train_time)
        all_test_times.append(test_time_per_image)
        all_fps.append(fps)
        all_vram.append(vram_used)

        print(f"🎯 Test Results - Loss: {test_loss:.4f}, "
              f"Acc: {test_acc*100:.2f}%, Bal Acc: {test_bal_acc*100:.2f}%, AUC: {test_auc*100:.2f}%")
        print(f"🕒 Train time: {train_time:.2f}s | Test time/img: {test_time_per_image*1000:.2f} ms | "
              f"FPS: {fps:.2f} | VRAM: {vram_used:.1f} MB")

        # === Save logs per run ===
        log_path = f"{save_dir}/log_{dataset_name}_{model_name}_patch{patch_size}_run{run+1}.csv"

        # === Save test probabilities ===
        prob_path = f"{save_dir}/test_predictions_{dataset_name}_{model_name}_patch{patch_size}_run{run+1}.csv"

        # === Plot curves ===
        plot_path = f"{save_dir}/curves_{dataset_name}_{model_name}_patch{patch_size}_run{run+1}.png"

        save_logs(history, log_path)
        save_probs(dataset_name, test_probs, num_classes, prob_path)
        plot_curves(history, plot_path)
        

    # === Aggregate results ===
    all_results = np.array(all_results)
    mean, std = all_results.mean(axis=0), all_results.std(axis=0)
    train_time_mean, train_time_std = np.mean(all_train_times), np.std(all_train_times)
    test_time_mean, test_time_std = np.mean(all_test_times), np.std(all_test_times)
    fps_mean, fps_std = np.mean(all_fps), np.std(all_fps)
    vram_mean, vram_std = np.mean(all_vram), np.std(all_vram)

    print(f"\n=== Final Summary ({robustness} runs) ===")
    print(f"Loss: {mean[0]:.2f} ± {std[0]:.2f}, Acc: {mean[1]*100:.2f}% ± {std[1]*100:.2f}%, "
          f"Bal Acc: {mean[2]*100:.2f}% ± {std[2]*100:.2f}%, AUC: {mean[3]*100:.2f}% ± {std[3]*100:.2f}%")
    print(f"Train Time: {train_time_mean:.2f} ± {train_time_std:.2f}s, "
          f"Test Time/img: {test_time_mean*1000:.2f} ± {test_time_std*1000:.2f}ms, "
          f"FPS: {fps_mean:.2f} ± {fps_std:.2f}, VRAM: {vram_mean:.1f} ± {vram_std:.1f} MB")

    # === Save extended summary CSV ===
    summary_file = f"{save_dir}/summary_results_{dataset_name}.csv"
    header = [
        "dataset", "model", "patch_size",
        "test_loss_mean", "test_loss_std",
        "test_acc_mean", "test_acc_std",
        "test_bal_acc_mean", "test_bal_acc_std",
        "test_auc_mean", "test_auc_std",
        "train_time_mean", "train_time_std",
        "test_time_per_image_mean", "test_time_per_image_std",
        "fps_mean", "fps_std",
        "vram_mb_mean", "vram_mb_std"
    ]

    row = [
        dataset, config["model_size"], patch_size,
        round(mean[0], 2), round(std[0], 2),
        round(mean[1]*100, 2), round(std[1]*100, 2),
        round(mean[2]*100, 2), round(std[2]*100, 2),
        round(mean[3]*100, 2), round(std[3]*100, 2),
        round(train_time_mean, 2), round(train_time_std, 2),
        round(test_time_mean*1000, 2), round(test_time_std*1000, 2),
        round(fps_mean, 2), round(fps_std, 2),
        round(vram_mean, 2), round(vram_std, 2)
    ]

    save_extended_summary(summary_file, header, row)

    return mean
