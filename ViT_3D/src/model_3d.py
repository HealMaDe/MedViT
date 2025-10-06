import timm
import torch.nn as nn
import torch

# -------------------------
# ViT-3D with Inflation Method
# -------------------------
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

        # Proper 3D convolution for patch embedding
        self.proj = nn.Conv3d(in_chans, embed_dim,
                            kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViT3D(nn.Module):
    """
    3D Vision Transformer with inflation method
    """
    def __init__(self, model_size="tiny", img_size=28, patch_size=14,
                 num_classes=2, pretrained=True, inflate_method="repeat",
                device="cpu"):
        super().__init__()

        self.device = device
        # Load pretrained 2D ViT (ImageNet Weights)
        if model_size == "tiny":
            self.vit_2d = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        elif model_size == "small":
            self.vit_2d = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
        elif model_size == "base":
            self.vit_2d = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

        # Move vit_2d to device first
        self.vit_2d = self.vit_2d.to(self.device)

        # Get ViT parameters
        embed_dim = self.vit_2d.embed_dim

        # Create proper 3D patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        ).to(self.device)

        # Inflate weights using specified method (we used the repeat method in thsi project)
        self._inflate_patch_embedding(patch_size, inflate_method)

        # Use ViT components
        self.cls_token = self.vit_2d.cls_token
        self.pos_drop = self.vit_2d.pos_drop
        self.blocks = self.vit_2d.blocks
        self.norm = self.vit_2d.norm

        # Initialize positional embeddings for 3D
        self.pos_embed = self._init_pos_embed_3d()

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes).to(self.device)
                    
        # Initialize head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        print(f"✅ Loaded ImageNet-pretrained {model_size} ViT-3D")
        print(f"✅ Patch size: {patch_size}, Inflation: {inflate_method}")

    def _inflate_patch_embedding(self, patch_size, method):
        """Convert 2D weights to 3D using specified method"""
        conv2d = self.vit_2d.patch_embed.proj
        weight_2d = conv2d.weight.data

        if method == "repeat":
            # I3D method: repeat along depth dimension
            depth = patch_size
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1)
            weight_3d = weight_3d / depth  # I3D normalization

        else:
            raise ValueError(f"Unknown inflation method: {method}")

        # Apply inflated weights
        with torch.no_grad():
            self.patch_embed.proj.weight.data = weight_3d.to(self.device)
            if conv2d.bias is not None:
                self.patch_embed.proj.bias.data = conv2d.bias.data.clone().to(self.device)

    def _init_pos_embed_3d(self):
          """Interpolate 2D positional embeddings into 3D space."""
          pos_embed_2d = self.vit_2d.pos_embed.to(self.device)  # (1, N+1, D)
          cls_embed = pos_embed_2d[:, 0:1, :]  # keep CLS as is
          patch_embed_2d = pos_embed_2d[:, 1:, :]  # (1, N, D)

          # Original 2D grid size
          num_patches_2d = patch_embed_2d.shape[1]
          grid_size_2d = int(num_patches_2d ** 0.5)
          patch_embed_2d = patch_embed_2d.reshape(1, grid_size_2d, grid_size_2d, -1)  # (1, H, W, D)

          # Target 3D grid size
          D, H, W = self.patch_embed.grid_size  # (depth, height, width)

          # Expand to fake depth dim -> (1, D_model, 1, H, W)
          patch_embed_2d = patch_embed_2d.permute(0, 3, 1, 2).unsqueeze(2)

          # Interpolate into (D, H, W)
          patch_embed_3d = torch.nn.functional.interpolate(
              patch_embed_2d,
              size=(D, H, W),
              mode="trilinear",
              align_corners=False
          )  # (1, D_model, D, H, W)

          # Reshape to (1, D*H*W, D_model)
          patch_embed_3d = patch_embed_3d.permute(0, 2, 3, 4, 1).reshape(1, D * H * W, -1)

          # Combine CLS + patches
          return nn.Parameter(torch.cat([cls_embed, patch_embed_3d], dim=1))

    def forward(self, x):
        # x: (B, 3, D, H, W)
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        current_seq_length = x.shape[1]
        pos_embed = self.pos_embed[:, :current_seq_length, :]
        x = x + pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)

        return x
