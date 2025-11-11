import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding that properly handles 3D input"""
    def __init__(self, img_size=28, patch_size=14, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = (img_size, img_size, img_size)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
            img_size // patch_size
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


class ViT3D(nn.Module):
    """
    3D Vision Transformer with inflation method.
    Supports return_attention=True for interpretability.
    """
    def __init__(self, model_size="tiny", img_size=28, patch_size=14,
                 num_classes=2, pretrained=True, inflate_method="repeat",
                 device="cpu"):
        super().__init__()
        self.device = device

        # ---------- Load pretrained 2D ViT (ImageNet) ----------
        if model_size == "tiny":
            self.vit_2d = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        elif model_size == "small":
            self.vit_2d = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
        elif model_size == "base":
            self.vit_2d = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        self.vit_2d = self.vit_2d.to(self.device)
        embed_dim = self.vit_2d.embed_dim

        # ---------- 3D Patch Embedding ----------
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        ).to(self.device)

        # ---------- Inflate 2D → 3D weights ----------
        self._inflate_patch_embedding(patch_size, inflate_method)

        # ---------- Re-use ViT components ----------
        self.cls_token = self.vit_2d.cls_token
        self.pos_drop = self.vit_2d.pos_drop
        self.blocks = self.vit_2d.blocks
        self.norm = self.vit_2d.norm

        # ---------- 3D Positional Embeddings (interpolated) ----------
        self.pos_embed = self._init_pos_embed_3d()

        # ---------- Classification head ----------
        self.head = nn.Linear(embed_dim, num_classes).to(self.device)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        print(f"Loaded ImageNet-pretrained {model_size} ViT-3D")
        print(f"Patch size: {patch_size}, Inflation: {inflate_method}")

    # --------------------------------------------------------------------- #
    #                     PATCH EMBEDDING INFLATION                        #
    # --------------------------------------------------------------------- #
    def _inflate_patch_embedding(self, patch_size, method):
        conv2d = self.vit_2d.patch_embed.proj
        weight_2d = conv2d.weight.data               # (embed_dim, in_chans, 16, 16)
        embed_dim, in_chans, src_h, src_w = weight_2d.shape
        assert src_h == src_w == 16, "Pretrained ViT must use patch 16"

        target_d = target_h = target_w = patch_size

        if method == "repeat":
            # ---- Step 1: Resize spatial dimensions (16×16 → P×P) ----
            weight_flat = weight_2d.view(embed_dim * in_chans, 1, src_h, src_w)
            weight_spatial = F.interpolate(
                weight_flat,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )                                            # (E*C, 1, P, P)
            weight_spatial = weight_spatial.view(embed_dim, in_chans, target_h, target_w)

            # ---- Step 2: Inflate depth (repeat along D) ----
            weight_3d = weight_spatial.unsqueeze(2)      # (E, C, 1, P, P)
            if target_d > 1:
                weight_3d = weight_3d.repeat(1, 1, target_d, 1, 1) / target_d
            # else: depth=1 → keep as-is

        elif method == "center_crop_or_pad":
            # Alternative: crop/pad 16×16 → P×P, then repeat depth
            w = weight_2d.view(embed_dim * in_chans, src_h, src_w)
            if patch_size > 16:
                pad = (patch_size - 16) // 2
                w = F.pad(w, (pad, pad, pad, pad))
                if (patch_size - 16) % 2:
                    w = F.pad(w, (0, 1, 0, 1))
            else:
                crop = (16 - patch_size) // 2
                w = w[:, crop:crop + patch_size, crop:crop + patch_size]

            weight_spatial = w.view(embed_dim, in_chans, patch_size, patch_size)
            weight_3d = weight_spatial.unsqueeze(2)
            weight_3d = weight_3d.expand(-1, -1, patch_size, -1, -1) / patch_size

        else:
            raise ValueError(f"Unknown inflation method: {method}")

        weight_3d = weight_3d.contiguous()

        # ---- Assign to 3D conv ----
        with torch.no_grad():
            self.patch_embed.proj.weight.copy_(weight_3d.to(self.device))
            if conv2d.bias is not None:
                self.patch_embed.proj.bias.copy_(conv2d.bias.data.to(self.device))

        print(f"   → 3D Conv weight shape: {self.patch_embed.proj.weight.shape}")

    # --------------------------------------------------------------------- #
    #                     POSITIONAL EMBEDDING 3D                           #
    # --------------------------------------------------------------------- #
    def _init_pos_embed_3d(self):
        pos_embed_2d = self.vit_2d.pos_embed.to(self.device)   # (1, N+1, D)
        cls_embed = pos_embed_2d[:, :1, :]                     # (1, 1, D)
        patch_embed_2d = pos_embed_2d[:, 1:, :]               # (1, 196, D)

        grid_2d = int(patch_embed_2d.shape[1] ** 0.5)
        assert grid_2d * grid_2d == patch_embed_2d.shape[1], "Non-square 2D grid!"

        # (1, H, W, D) → (1, D, 1, H, W)
        patch_embed_2d = patch_embed_2d.view(1, grid_2d, grid_2d, -1)
        patch_embed_2d = patch_embed_2d.permute(0, 3, 1, 2).unsqueeze(2)

        D, H, W = self.patch_embed.grid_size

        # Interpolate to target 3D grid
        patch_embed_3d = F.interpolate(
            patch_embed_2d,
            size=(D, H, W),
            mode='trilinear',
            align_corners=False
        )

        patch_embed_3d = patch_embed_3d.permute(0, 2, 3, 4, 1).reshape(1, D * H * W, -1)
        pos_embed_3d = torch.cat([cls_embed, patch_embed_3d], dim=1)

        return nn.Parameter(pos_embed_3d)

    # --------------------------------------------------------------------- #
    #                               FORWARD                                 #
    # --------------------------------------------------------------------- #
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)                         # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, N+1, D)

        # Truncate / pad pos_embed if needed
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)

        attentions = []
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if return_attention:
            return logits, attentions
        return logits
