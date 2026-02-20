# Effect of Patch Size on Fine-tuning Vision Transformers In 2D and 3D Medical Image Classification

This repository contains fine-tuning of ViTs for both 2D and 3D medical image classification tasks. The code supports customizable patch sizes and models (tiny, small, base, etc.).

## Overview

The project provides two separate pipelines:
- **2D ViT**: For standard 2D medical image datasets (e.g., BreastMNIST)
- **3D ViT**: For volumetric medical image datasets (e.g., VesselMNIST3D)

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/HealMaDe/MedViT
cd MedViT

# Install dependencies
pip install -r requirements.txt
```
### Experiments

All experiment settings are controlled **directly inside the main scripts**

Open `main.py` and set the desired variables

```python
# --- User-configurable parameters ---
dataset = "breastmnist"            # Dataset name
img_size = 28                      # Input image size
patch_sizes = [28, 14, 7]          # Patch sizes to evaluate
models = ["vit_base_patch16_224"]  # ViT backbone (timm)
robustness = 3                     # Number of repeated runs
```

* The experiment will automatically run **one experiment per patch size**
* `robustness` controls how many times each configuration is repeated

Run the experiment:

```bash
python main.py
```


