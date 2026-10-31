# fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    def __init__(self, models_dict, num_classes, fusion_dim=None):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)

        self.feature_dims = {}
        for patch_size, model in self.models.items():
            if hasattr(model, 'embed_dim'):
                self.feature_dims[patch_size] = model.embed_dim
            else:
                self.feature_dims[patch_size] = 768

        total_dim = sum(self.feature_dims.values())
        fusion_dim = fusion_dim or total_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)  # logits, no softmax

        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    def train(self, mode=True):
        super().train(mode)
        for model in self.models.values():
            model.eval()
        return self

    def forward(self, x):
        features = []
        for patch_size, model in self.models.items():
            if hasattr(model, 'forward_features'):
                with torch.no_grad():
                    feat = model.forward_features(x)
                    if feat.dim() == 3:
                        feat = feat[:, 0]  # CLS token
            else:
                raise ValueError(f"Model {model.__class__.__name__} doesn't have forward_features")
            features.append(feat)

        fused = torch.cat(features, dim=1)
        fused = self.fusion_layer(fused)
        output = self.classifier(fused)  # logits
        return output


class CrossAttentionFusion(nn.Module):
    def __init__(self, models_dict, num_classes, feature_dim=256, num_heads=4):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)

        self.feature_dims = {}
        for patch_size, model in self.models.items():
            if hasattr(model, 'embed_dim'):
                self.feature_dims[patch_size] = model.embed_dim
            else:
                self.feature_dims[patch_size] = 768

        self.projections = nn.ModuleDict({
            str(patch_size): nn.Sequential(
                nn.Linear(dim, feature_dim),
                nn.LayerNorm(feature_dim)
            )
            for patch_size, dim in self.feature_dims.items()
        })

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)  # logits
        )

        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        self._init_weights()

    def train(self, mode=True):
        
        super().train(mode)
        for model in self.models.values():
            model.eval()
        return self

    def _init_weights(self):
        new_submodules = [self.projections, self.cross_attn, self.fusion_layer, self.classifier]
        for parent in new_submodules:
            for module in parent.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        projected_features = []
        for patch_size, model in self.models.items():
            if hasattr(model, 'forward_features'):
                with torch.no_grad():
                    feat = model.forward_features(x)
                    if feat.dim() == 3:
                        feat = feat[:, 0]
            else:
                raise ValueError(f"Model {model.__class__.__name__} doesn't have forward_features")

            feat = self.projections[str(patch_size)](feat)
            projected_features.append(feat)

        
        stacked = torch.stack(projected_features, dim=1)

        query = stacked
        key = stacked
        value = stacked

        attn_output, _ = self.cross_attn(query, key, value)
        fused = attn_output.mean(dim=1)  

        fused = fused + self.fusion_layer(fused)
        output = self.classifier(fused)  # logits
        return output



class LateFusionModel(nn.Module):
    def __init__(self, models_dict, num_classes, weighting="scalar"):
        super().__init__()
        assert weighting in ("scalar", "per_class")
        self.weighting = weighting
        self.models = nn.ModuleDict(models_dict)
        n = len(models_dict)

        if weighting == "scalar":
            self.branch_logits_weight = nn.Parameter(torch.zeros(n))  
        else:
            self.branch_logits_weight = nn.Parameter(torch.zeros(n, num_classes))

        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    def train(self, mode=True):
        super().train(mode)
        for model in self.models.values():
            model.eval()
        return self

    def get_branch_weights(self):
        
        return torch.softmax(self.branch_logits_weight, dim=0)

    def forward(self, x, return_log_probs=True):

        probs_list = []
        for model in self.models.values():
            with torch.no_grad():
                logits = model(x)  
                probs_list.append(torch.softmax(logits, dim=1))

        probs_stack = torch.stack(probs_list, dim=1)  
        weights = self.get_branch_weights() 

        if self.weighting == "scalar":
            weights = weights.view(1, -1, 1)  # (1, n, 1)
        else:
            weights = weights.unsqueeze(0)  # (1, n, num_classes)

        fused_probs = (probs_stack * weights).sum(dim=1)  

        if return_log_probs:
            return torch.log(fused_probs.clamp_min(1e-8))  
        return fused_probs


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
