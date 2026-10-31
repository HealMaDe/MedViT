# calculate_gflops_vit3d.py

import gc
import csv
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from model_3d import ViT3D


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET = "vesselmnist3d"

DATA_DIR = Path(
    "/mnt/18T/healmade/"
    "vit_patch/"
    "MedViT/"
    "ViT_3D/"
    "data"
)


# ------------------------------------------------------------
# Model size
#
# Options:
#   "tiny"
#   "small"
#   "base"
# ------------------------------------------------------------

MODEL_SIZE = "base"


# ------------------------------------------------------------
# 3D image size
# ------------------------------------------------------------

IMG_SIZE = 28


# ------------------------------------------------------------
# Cubic patch sizes
#
# Patch kernel:
#   patch=1  -> (1, 1, 1)
#   patch=2  -> (2, 2, 2)
#   patch=4  -> (4, 4, 4)
#   ...
# ------------------------------------------------------------

PATCH_SIZES = [
    1,
    2,
    4,
    7,
    14,
    28,
]


# ------------------------------------------------------------
# Inflation
# ------------------------------------------------------------

INFLATE_METHOD = "repeat"


# ------------------------------------------------------------
# Pretrained
#
# GFLOPs do NOT depend on trained weight values.
#
# False is faster and avoids downloading/loading weights.
# Architecture and FLOPs remain the same.
# ------------------------------------------------------------

PRETRAINED = False


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_CSV = Path(
    f"gflops_vit3d_{DATASET}_{MODEL_SIZE}.csv"
)


# ============================================================
# GET NUMBER OF CLASSES
# ============================================================

def get_num_classes(dataset):

    npz_path = DATA_DIR / f"{dataset}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{npz_path}"
        )

    data = np.load(npz_path)

    all_labels = np.concatenate([
        data["train_labels"].reshape(-1),
        data["val_labels"].reshape(-1),
        data["test_labels"].reshape(-1),
    ])

    unique_classes = np.unique(all_labels)

    print(
        f"Detected classes: {unique_classes.tolist()}"
    )

    return len(unique_classes)


# ============================================================
# CALCULATE GFLOPs
# ============================================================

def calculate_gflops(
    model,
    img_size,
    device,
):

    model.eval()

    # --------------------------------------------------------
    # EXACTLY ONE 3D VOLUME
    #
    # Shape:
    #   B, C, D, H, W
    #
    # Batch size = 1 is important.
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        img_size,
        img_size,
        img_size,
        device=device,
    )

    # --------------------------------------------------------
    # fvcore analysis
    # --------------------------------------------------------

    with torch.no_grad():

        flop_counter = FlopCountAnalysis(
            model,
            dummy_input,
        )

        # Suppress huge warning messages.
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)

        total_flops = flop_counter.total()

        unsupported_ops = flop_counter.unsupported_ops()

    gflops = total_flops / 1e9

    return (
        total_flops,
        gflops,
        unsupported_ops,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("ViT-3D GFLOPs — CUBIC PATCH EMBEDDING")
    print("=" * 100)

    print(f"Dataset:       {DATASET}")
    print(f"Model size:    {MODEL_SIZE}")
    print(f"Image size:    {IMG_SIZE}³")
    print(f"Device:        {DEVICE}")
    print(f"Batch size:    1")
    print(f"Pretrained:    {PRETRAINED}")
    print(f"Inflation:     {INFLATE_METHOD}")

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = get_num_classes(
        DATASET
    )

    print(f"Num classes:   {num_classes}")

    print("=" * 100)

    results = []

    # ========================================================
    # PATCH SIZE LOOP
    # ========================================================

    for patch_size in PATCH_SIZES:

        print()
        print("-" * 100)

        print(
            f"ViT-{MODEL_SIZE} | "
            f"image={IMG_SIZE}³ | "
            f"patch={patch_size}³"
        )

        print("-" * 100)

        # ----------------------------------------------------
        # Check divisibility
        # ----------------------------------------------------

        if IMG_SIZE % patch_size != 0:

            print(
                f"SKIPPED: {IMG_SIZE} is not divisible "
                f"by patch size {patch_size}"
            )

            continue

        model = None

        try:

            # =================================================
            # CREATE EXACT ViT3D ARCHITECTURE
            # =================================================

            model = ViT3D(
                model_size=MODEL_SIZE,
                img_size=IMG_SIZE,
                patch_size=patch_size,
                num_classes=num_classes,
                pretrained=PRETRAINED,
                inflate_method=INFLATE_METHOD,
                device=DEVICE,
            )

            model = model.to(DEVICE)
            model.eval()

            # =================================================
            # ARCHITECTURE INFORMATION
            # =================================================

            grid_size = (
                IMG_SIZE // patch_size
            )

            grid_size_3d = (
                grid_size,
                grid_size,
                grid_size,
            )

            num_patch_tokens = (
                grid_size ** 3
            )

            # +1 CLS token
            sequence_length = (
                num_patch_tokens + 1
            )

            patch_kernel = (
                patch_size,
                patch_size,
                patch_size,
            )

            # =================================================
            # PARAMETER COUNT
            # =================================================

            total_params = sum(
                p.numel()
                for p in model.parameters()
            )

            # =================================================
            # GFLOPs
            # =================================================

            (
                total_flops,
                gflops,
                unsupported_ops,

            ) = calculate_gflops(
                model=model,
                img_size=IMG_SIZE,
                device=DEVICE,
            )

            # =================================================
            # PRINT
            # =================================================

            print()

            print(
                f"Patch kernel:       "
                f"{patch_kernel}"
            )

            print(
                f"Patch grid:         "
                f"{grid_size_3d}"
            )

            print(
                f"Patch tokens:       "
                f"{num_patch_tokens:,}"
            )

            print(
                f"Sequence length:    "
                f"{sequence_length:,}"
            )

            print(
                f"Parameters:         "
                f"{total_params / 1e6:.6f} M"
            )

            print(
                f"FLOPs:              "
                f"{total_flops:,.0f}"
            )

            print(
                f"GFLOPs / volume:    "
                f"{gflops:.6f}"
            )

            # =================================================
            # UNSUPPORTED OPS
            # =================================================

            if unsupported_ops:

                print()
                print(
                    "WARNING: fvcore unsupported operators:"
                )

                for op, count in unsupported_ops.items():

                    print(
                        f"    {op}: {count}"
                    )

            else:

                print()
                print(
                    "Unsupported operators: none"
                )

            # =================================================
            # SAVE
            # =================================================

            results.append({

                "dataset":
                    DATASET,

                "model_size":
                    MODEL_SIZE,

                "img_size":
                    IMG_SIZE,

                "input_shape":
                    f"3x{IMG_SIZE}x{IMG_SIZE}x{IMG_SIZE}",

                "patch_size":
                    patch_size,

                "patch_kernel":
                    str(patch_kernel),

                "grid_size_3d":
                    str(grid_size_3d),

                "num_patch_tokens":
                    num_patch_tokens,

                "sequence_length_with_cls":
                    sequence_length,

                "num_classes":
                    num_classes,

                "num_params_M":
                    round(
                        total_params / 1e6,
                        6,
                    ),

                "total_flops":
                    int(total_flops),

                "gflops_per_volume":
                    round(
                        gflops,
                        6,
                    ),

                "unsupported_ops":
                    str(
                        dict(unsupported_ops)
                    ),

            })

        except RuntimeError as e:

            # =================================================
            # Usually CUDA OOM for patch=1
            # =================================================

            print()
            print(
                f"RUNTIME ERROR: {e}"
            )

            results.append({

                "dataset":
                    DATASET,

                "model_size":
                    MODEL_SIZE,

                "img_size":
                    IMG_SIZE,

                "patch_size":
                    patch_size,

                "error":
                    str(e),

            })

        except Exception as e:

            print()
            print(
                f"FAILED: {e}"
            )

            results.append({

                "dataset":
                    DATASET,

                "model_size":
                    MODEL_SIZE,

                "img_size":
                    IMG_SIZE,

                "patch_size":
                    patch_size,

                "error":
                    str(e),

            })

        finally:

            # =================================================
            # CLEAN MEMORY
            # =================================================

            if model is not None:

                del model

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # ========================================================
    # SAVE CSV
    # ========================================================

    if not results:

        print(
            "\nNo results generated."
        )

        return

    fieldnames = []

    for row in results:

        for key in row:

            if key not in fieldnames:

                fieldnames.append(key)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL GFLOPs")
    print("=" * 100)

    for row in results:

        if "gflops_per_volume" in row:

            print(
                f"ViT-{MODEL_SIZE:5s} | "
                f"img={IMG_SIZE:2d}³ | "
                f"patch={row['patch_size']:2d}³ | "
                f"tokens={row['num_patch_tokens']:6d} | "
                f"{row['gflops_per_volume']:12.6f} GFLOPs/volume"
            )

        else:

            print(
                f"ViT-{MODEL_SIZE:5s} | "
                f"img={IMG_SIZE:2d}³ | "
                f"patch={row['patch_size']:2d}³ | "
                f"FAILED"
            )

    print("=" * 100)

    print(
        f"\nSaved to:\n{OUTPUT_CSV}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
