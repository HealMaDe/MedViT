import csv
import os
import torch
import matplotlib.pyplot as plt
import gc

def save_logs(history, save_path):
  with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(history.keys())
            writer.writerows(zip(*history.values()))

def save_probs(dataset_name, test_probs, num_classes, prob_path):
  with open(prob_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["dataset", "true_label"] + [f"prob_class_{i}" for i in range(num_classes)]
            writer.writerow(header)
            for label, probs in test_probs:
                writer.writerow([dataset_name, label] + probs.tolist())


def plot_curves(history,plot_path):
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 4, 1); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val")
        plt.title("Loss"); plt.legend()
        plt.subplot(1, 4, 2); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val")
        plt.title("Accuracy"); plt.legend()
        plt.subplot(1, 4, 3); plt.plot(history["val_bal_acc"], label="Val Bal Acc", color='orange')
        plt.title("Balanced Accuracy"); plt.legend()
        plt.subplot(1, 4, 4); plt.plot(history["val_auc"], label="Val AUC")
        plt.title("Validation AUC"); plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        gc.collect()
        torch.cuda.empty_cache()

def save_extended_summary(summary_file, header, row):
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
