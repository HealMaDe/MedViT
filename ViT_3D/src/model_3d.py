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



def _swin3d_stage_config(img_size, patch_size, base_depths=(2, 2, 6, 2),
                          base_num_heads=(3, 6, 12, 24), base_window=7):
    if img_size % patch_size != 0:
        raise ValueError(f"img_size={img_size} bar patch_size={patch_size} bakhsh-pazir nist")
    grid = img_size // patch_size
    max_merges = len(base_depths) - 1
    k, r = 0, grid
    while k < max_merges and r % 2 == 0:
        r //= 2
        k += 1
    depths = base_depths[: k + 1]
    num_heads = base_num_heads[: k + 1]
    coarsest_res = grid // (2 ** k)
    window_size = 1
    for w in range(min(base_window, coarsest_res), 0, -1):
        if coarsest_res % w == 0:
            window_size = w
            break
    return depths, num_heads, window_size, grid, k + 1


def _inflate_relative_position_bias(rpb_2d, window_size_2d, window_size_3d):
    wh2, ww2 = window_size_2d, window_size_2d
    wd3, wh3, ww3 = window_size_3d, window_size_3d, window_size_3d
    heads = rpb_2d.shape[1]

    rpb_2d_grid = rpb_2d.reshape(2 * wh2 - 1, 2 * ww2 - 1, heads)

    if (2 * wh3 - 1, 2 * ww3 - 1) != (2 * wh2 - 1, 2 * ww2 - 1):
        rpb_2d_grid = F.interpolate(
            rpb_2d_grid.permute(2, 0, 1).unsqueeze(0),
            size=(2 * wh3 - 1, 2 * ww3 - 1), mode="bicubic", align_corners=False
        ).squeeze(0).permute(1, 2, 0)

    rpb_3d_grid = rpb_2d_grid.unsqueeze(0).repeat(2 * wd3 - 1, 1, 1, 1)
    return rpb_3d_grid.reshape(-1, heads)


class PatchEmbed3DSwin(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=96):
        super().__init__()
        self.grid_size = (img_size // patch_size,) * 3
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        D, H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), (D, H, W)


def window_partition_3d(x, D, H, W, window_size):
    B, N, C = x.shape
    ws = window_size
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, ws * ws * ws, C)
    return windows


def window_reverse_3d(windows, D, H, W, window_size, B):
    ws = window_size
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D * H * W, -1)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), torch.arange(ws), indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 2] += ws - 1
        rel_coords[:, :, 0] *= (2 * ws - 1) * (2 * ws - 1)
        rel_coords[:, :, 1] *= (2 * ws - 1)
        relative_position_index = rel_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, D, H, W):
        B = x.shape[0]
        shortcut = x
        x = self.norm1(x)
        windows = window_partition_3d(x, D, H, W, self.window_size)
        windows = self.attn(windows)
        x = window_reverse_3d(windows, D, H, W, self.window_size, B)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.view(B, D, H, W, C)
        pieces = [x[:, i::2, j::2, k::2, :] for i in range(2) for j in range(2) for k in range(2)]
        x = torch.cat(pieces, dim=-1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        return self.reduction(x), (D // 2, H // 2, W // 2)


class Swin3D(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", img_size=28, patch_size=4,
                 num_classes=2, pretrained=True, device="cpu"):
        super().__init__()
        self.device = device
        ref2d = timm.create_model(model_name, pretrained=pretrained)

        depths, num_heads, window_size, grid, n_stages = _swin3d_stage_config(
            img_size, patch_size
        )
        print(f"[Swin3D cfg] patch={patch_size} img={img_size} -> grid={grid}^3, "
              f"stages={n_stages}, depths={depths}, window={window_size}")

        embed_dim = ref2d.embed_dim if hasattr(ref2d, "embed_dim") else ref2d.layers[0].blocks[0].norm1.normalized_shape[0]

        self.patch_embed = PatchEmbed3DSwin(img_size, patch_size, 3, embed_dim)
        self._inflate_patch_embed(ref2d, patch_size)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        for stage_idx in range(n_stages):
            blocks = nn.ModuleList([
                SwinBlock3D(dim, num_heads[stage_idx], window_size)
                for _ in range(depths[stage_idx])
            ])
            self.stages.append(blocks)
            self._inflate_stage(blocks, ref2d, stage_idx, window_size)
            if stage_idx < n_stages - 1:
                merge = PatchMerging3D(dim)
                self.merges.append(merge)
                dim *= 2
            else:
                self.merges.append(None)

        self.norm = nn.LayerNorm(dim)
        self.embed_dim = dim
        self.num_features = dim

        self.head = nn.Linear(dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        del ref2d

        self.info = self._build_info(model_name, patch_size, img_size, pretrained,
                                      depths, num_heads, window_size, n_stages, dim)
        print(f"✅ Swin3D ready: {n_stages} stages, final dim={dim}")

    def _build_info(self, model_name, patch_size, img_size, pretrained,
                     depths, num_heads, window_size, n_stages, final_dim):
        total_params, trainable_params = _count_params(self)
        head_params = sum(p.numel() for p in self.head.parameters())

        random_params = 0
        for m in self.merges:
            if m is not None:
                random_params += sum(p.numel() for p in m.parameters())

        backbone_params = total_params - head_params
        backbone_pretrained_derived_ratio = (
            1.0 - (random_params / backbone_params) if (pretrained and backbone_params > 0) else 0.0
        )

        return {
            "family": "swin",
            "model_name": model_name,
            "pretrained": pretrained,
            "patch_size": patch_size,
            "img_size": img_size,
            "num_stages": n_stages,
            "depth_per_stage": str(list(depths)),
            "depth_total": sum(depths),
            "num_heads_per_stage": str(list(num_heads)),
            "window_size": window_size,
            "embed_dim_stage1": self.patch_embed.proj.out_channels,
            "final_dim": final_dim,
            "num_patch_merges": max(n_stages - 1, 0),
            "patch_embed_source": f"inflated_repeat" if pretrained else "random_init_no_pretrain",
            "rel_pos_bias_source": "inflated_repeat" if pretrained else "random_init_no_pretrain",
            "patch_merging_reduction_source": "random_reinit (shape 8C->2C, no 2D equivalent)",
            "backbone_pretrained_derived_ratio": round(backbone_pretrained_derived_ratio, 3),
            "num_params_total": total_params,
            "num_params_backbone": backbone_params,
            "num_params_head": head_params,
            "num_params_random_reinit": random_params,
            "num_params_M": round(total_params / 1e6, 2),
        }

    def _inflate_patch_embed(self, ref2d, patch_size):
        w2d = ref2d.patch_embed.proj.weight.data
        native_p = w2d.shape[-1]
        if native_p != patch_size:
            w2d = F.interpolate(w2d, size=(patch_size, patch_size),
                                 mode="bicubic", align_corners=False)
        w3d = w2d.unsqueeze(2).repeat(1, 1, patch_size, 1, 1) / patch_size
        with torch.no_grad():
            self.patch_embed.proj.weight.data = w3d.clone()
            if ref2d.patch_embed.proj.bias is not None:
                self.patch_embed.proj.bias.data = ref2d.patch_embed.proj.bias.data.clone()
            if hasattr(ref2d.patch_embed, "norm") and isinstance(ref2d.patch_embed.norm, nn.LayerNorm):
                self.patch_embed.norm.load_state_dict(ref2d.patch_embed.norm.state_dict())

    def _inflate_stage(self, blocks3d, ref2d, stage_idx, window_size_3d):
        ref_blocks = ref2d.layers[stage_idx].blocks
        window_size_2d = ref_blocks[0].attn.window_size[0]
        for i, blk3d in enumerate(blocks3d):
            ref_blk = ref_blocks[i % len(ref_blocks)]
            with torch.no_grad():
                blk3d.norm1.load_state_dict(ref_blk.norm1.state_dict())
                blk3d.norm2.load_state_dict(ref_blk.norm2.state_dict())
                blk3d.attn.qkv.load_state_dict(ref_blk.attn.qkv.state_dict())
                blk3d.attn.proj.load_state_dict(ref_blk.attn.proj.state_dict())
                blk3d.mlp[0].load_state_dict(ref_blk.mlp.fc1.state_dict())
                blk3d.mlp[2].load_state_dict(ref_blk.mlp.fc2.state_dict())
                rpb_3d = _inflate_relative_position_bias(
                    ref_blk.attn.relative_position_bias_table.data,
                    window_size_2d, window_size_3d,
                )
                blk3d.attn.relative_position_bias_table.data = rpb_3d.clone()

    def forward_features(self, x):
        x, (D, H, W) = self.patch_embed(x)
        for stage_idx, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, D, H, W)
            if self.merges[stage_idx] is not None:
                x, (D, H, W) = self.merges[stage_idx](x, D, H, W)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)
