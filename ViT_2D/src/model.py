import timm
import torch.nn as nn
import torch

def get_model(model_name, num_classes, patch_size, img_size, device):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size
    )
    return model.to(device)
