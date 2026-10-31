import gc
import csv
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET = "organsmnist"

DATA_DIR = Path(
    "/mnt/18T/healmade/"
    "vit_patch/"
    "MedViT/"
    "ViT_2D/"
    "data"
)


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MODELS = [

    "vit_tiny_patch16_224",

    #"vit_small_patch16_224",

    #"swin_tiny_patch4_window7_224",

    #"swin_small_patch4_window7_224",

    #"flexivit_base.1200ep_in1k",

]


# ------------------------------------------------------------
# Image sizes
# ------------------------------------------------------------

IMAGE_SIZES = [

    28,

    # 28,

    # 224,

]


# ------------------------------------------------------------
# Patch sizes
# ------------------------------------------------------------

PATCH_SIZES = [

    1,
    2,
    4,
    7,
    #8,
    14,
    #16,
    28,
    #32,
    #64,
    #128,

]


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_CSV = Path()


# ============================================================
# GET NUMBER OF CLASSES
# ============================================================

def get_num_classes(dataset):

    npz_path = DATA_DIR / f"{dataset}.npz"

    if not npz_path.exists():

        raise FileNotFoundError(
            f"\nDataset file not found:\n"
            f"{npz_path}\n"
        )

    data = np.load(npz_path)

    labels = data["train_labels"].reshape(-1)

    unique_classes = np.unique(labels)

    num_classes = len(unique_classes)

    print(
        f"Detected classes: {unique_classes.tolist()}"
    )

    return num_classes


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
    # One RGB image
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        img_size,
        img_size,
        device=device,
    )

    # --------------------------------------------------------
    # FLOP analysis
    # --------------------------------------------------------

    with torch.no_grad():

        flop_analysis = FlopCountAnalysis(
            model,
            dummy_input,
        )

        # Prevent very large warning outputs
        flop_analysis.unsupported_ops_warnings(False)
        flop_analysis.uncalled_modules_warnings(False)

        total_flops = flop_analysis.total()

        unsupported_ops = flop_analysis.unsupported_ops()

    # --------------------------------------------------------
    # Convert FLOPs -> GFLOPs
    # --------------------------------------------------------

    gflops = total_flops / 1e9

    return gflops, unsupported_ops


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print("GFLOPs CALCULATION — INDIVIDUAL MODELS")
    print("=" * 100)

    print(
        f"Dataset:       {DATASET}"
    )

    print(
        f"Device:        {DEVICE}"
    )

    print(
        "Batch size:    1"
    )

    print(
        "Input channels: 3"
    )

    # --------------------------------------------------------
    # Determine number of classes
    # --------------------------------------------------------

    num_classes = get_num_classes(DATASET)

    print(
        f"Num classes:   {num_classes}"
    )

    print("=" * 100)

    results = []

    # ========================================================
    # LOOP THROUGH MODELS
    # ========================================================

    for model_name in MODELS:

        for img_size in IMAGE_SIZES:

            for patch_size in PATCH_SIZES:

                print("\n")
                print("-" * 100)

                print(
                    f"Model:       {model_name}"
                )

                print(
                    f"Image size:  {img_size}x{img_size}"
                )

                print(
                    f"Patch size:  {patch_size}"
                )

                print("-" * 100)

                # ------------------------------------------------
                # Basic divisibility check
                # ------------------------------------------------

                if img_size % patch_size != 0:

                    print(
                        "SKIPPED:"
                        f" image size {img_size}"
                        f" is not divisible by"
                        f" patch size {patch_size}."
                    )

                    continue

                model = None

                try:

                    # ============================================
                    # BUILD MODEL
                    # ============================================

                    model, arch_info = get_model(

                        model_name=model_name,

                        num_classes=num_classes,

                        patch_size=patch_size,

                        img_size=img_size,

                        device=DEVICE,

                    )

                    model.eval()

                    # ============================================
                    # CALCULATE GFLOPs
                    # ============================================

                    gflops, unsupported_ops = (
                        calculate_gflops(
                            model=model,
                            img_size=img_size,
                            device=DEVICE,
                        )
                    )

                    # ============================================
                    # ARCHITECTURE INFORMATION
                    # ============================================

                    grid_size = (
                        img_size // patch_size
                    )

                    patch_tokens = (
                        grid_size * grid_size
                    )

                    total_params = sum(
                        p.numel()
                        for p in model.parameters()
                    )

                    # ============================================
                    # PRINT RESULT
                    # ============================================

                    print()

                    print(
                        f"Grid:             "
                        f"{grid_size}x{grid_size}"
                    )

                    print(
                        f"Patch tokens:     "
                        f"{patch_tokens}"
                    )

                    print(
                        f"Parameters:       "
                        f"{total_params / 1e6:.4f} M"
                    )

                    print(
                        f"GFLOPs/image:     "
                        f"{gflops:.6f}"
                    )

                    # ============================================
                    # UNSUPPORTED OPERATORS
                    # ============================================

                    if unsupported_ops:

                        print(
                            "\nUnsupported fvcore operators:"
                        )

                        for op, count in (
                            unsupported_ops.items()
                        ):

                            print(
                                f"    {op}: {count}"
                            )

                    else:

                        print(
                            "\nUnsupported fvcore operators: none"
                        )

                    # ============================================
                    # SAVE RESULT
                    # ============================================

                    results.append({

                        "dataset":
                            DATASET,

                        "num_classes":
                            num_classes,

                        "model_name":
                            model_name,

                        "family":
                            arch_info.get(
                                "family"
                            ),

                        "img_size":
                            img_size,

                        "patch_size":
                            patch_size,

                        "grid_size":
                            arch_info.get(
                                "grid_size_stage1",
                                grid_size,
                            ),

                        "num_tokens":
                            arch_info.get(
                                "num_tokens_stage1",
                                patch_tokens,
                            ),

                        "num_stages":
                            arch_info.get(
                                "num_stages"
                            ),

                        "depth_total":
                            arch_info.get(
                                "depth_total"
                            ),

                        "window_size":
                            arch_info.get(
                                "window_size"
                            ),

                        "num_params_M":
                            round(
                                total_params / 1e6,
                                6,
                            ),

                        "gflops_per_image":
                            round(
                                gflops,
                                6,
                            ),

                        "unsupported_ops":
                            str(
                                dict(
                                    unsupported_ops
                                )
                            ),

                    })

                except Exception as e:

                    print(
                        f"\nFAILED: {e}"
                    )

                    results.append({

                        "dataset":
                            DATASET,

                        "num_classes":
                            num_classes,

                        "model_name":
                            model_name,

                        "img_size":
                            img_size,

                        "patch_size":
                            patch_size,

                        "error":
                            str(e),

                    })

                finally:

                    # ============================================
                    # MEMORY CLEANUP
                    # ============================================

                    if model is not None:

                        del model

                    gc.collect()

                    if torch.cuda.is_available():

                        torch.cuda.empty_cache()

    # ========================================================
    # SAVE CSV
    # ========================================================

    if len(results) == 0:

        print(
            "\nNo results were generated."
        )

        return

    # --------------------------------------------------------
    # Collect every possible CSV column
    # --------------------------------------------------------

    fieldnames = []

    for row in results:

        for key in row.keys():

            if key not in fieldnames:

                fieldnames.append(key)

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

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

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL GFLOPs RESULTS")
    print("=" * 100)

    for row in results:

        if "gflops_per_image" in row:

            print(

                f"{row['model_name']:35s} | "

                f"img={row['img_size']:3d} | "

                f"patch={row['patch_size']:3d} | "

                f"{row['gflops_per_image']:12.6f} GFLOPs"

            )

        else:

            print(

                f"{row['model_name']:35s} | "

                f"img={row['img_size']:3d} | "

                f"patch={row['patch_size']:3d} | "

                f"FAILED"

            )

    print("=" * 100)

    print(
        f"\nResults saved to:\n{OUTPUT_CSV}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
