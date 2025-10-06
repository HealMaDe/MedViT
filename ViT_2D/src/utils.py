import csv
import os
import torch
import matplotlib.pyplot as plt

def save_logs(history, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

def save_test_predictions(predictions, save_path):
    """
    predictions: list of dicts, each with keys: 'dataset', 'true_label', 'probabilities'
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "true_label"] + [f"class_{i}_prob" for i in range(len(predictions[0]["probabilities"]))])
        for p in predictions:
            writer.writerow([p["dataset"], p["true_label"], *p["probabilities"]])

def plot_curves(history, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["val_auc"], label="Val AUC")
    plt.title("Validation AUC"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

