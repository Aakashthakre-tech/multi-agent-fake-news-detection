from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

class TemporalAttentionPooling(nn.Module):
    """
    Self-attention pooling over frame features in a video sequence.
    Given (B, T, D), computes attention scores (B, T, 1) and returns weighted sum (B, D).
    """
    def __init__(self, embed_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = self.attn(x)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled

class TemporalVideoClassifier(nn.Module):
    """
    TruthLensAI Video Classifier combining:
    1. Pretrained EfficientNet-B0 frame feature representations (1280-d).
    2. Temporal Aggregation:
       - 'hybrid': Temporal Attention Pooling + Mean Pooling + Max Pooling (original baseline)
       - 'projected_dynamics': Input LayerNorm + Linear Projection + Mean + Max + Temporal Difference Dynamics
       - 'mean_max': Mean + Max pooling
       - 'mean': Mean pooling
       - 'attention': Attention pooling
    3. Multi-layer Perceptron (MLP) binary classifier with Dropout and LayerNorm.
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 1280,
        hidden_dim: int = 256,
        mlp_dim: int = 256,
        dropout: float = 0.4,
        pooling_mode: str = "hybrid",
        proj_dim: Optional[int] = None,
        norm_type: str = "none",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling_mode = pooling_mode
        self.proj_dim = proj_dim
        self.norm_type = norm_type

        # Input Normalization
        if norm_type == "layernorm":
            self.input_norm = nn.LayerNorm(embed_dim)
        else:
            self.input_norm = nn.Identity()

        # Projection
        if proj_dim is not None and proj_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            working_dim = proj_dim
        else:
            self.projection = nn.Identity()
            working_dim = embed_dim

        if pooling_mode == "hybrid":
            self.attn_pool = TemporalAttentionPooling(embed_dim=working_dim, hidden_dim=hidden_dim)
            combined_dim = working_dim * 3
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, num_classes),
            )
        elif pooling_mode == "projected_dynamics":
            combined_dim = working_dim * 3
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
            )
        elif pooling_mode == "mean_max":
            combined_dim = working_dim * 2
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
            )
        elif pooling_mode in ("mean", "attention"):
            if pooling_mode == "attention":
                self.attn_pool = TemporalAttentionPooling(embed_dim=working_dim, hidden_dim=min(hidden_dim, 64))
            combined_dim = working_dim
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: Tensor of shape (B, T, 1280) containing extracted frame features.
        Returns:
            Logits of shape (B, num_classes).
        """
        if self.norm_type == "l2":
            x = nn.functional.normalize(frame_features, p=2, dim=-1)
        else:
            x = self.input_norm(frame_features)

        x = self.projection(x)

        if self.pooling_mode == "hybrid":
            attn_rep = self.attn_pool(x)
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            combined = torch.cat([attn_rep, mean_rep, max_rep], dim=1)
        elif self.pooling_mode == "projected_dynamics":
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            if x.size(1) > 1:
                diff = torch.abs(x[:, 1:, :] - x[:, :-1, :])
                diff_rep = torch.mean(diff, dim=1)
            else:
                diff_rep = torch.zeros_like(mean_rep)
            combined = torch.cat([mean_rep, max_rep, diff_rep], dim=1)
        elif self.pooling_mode == "mean_max":
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            combined = torch.cat([mean_rep, max_rep], dim=1)
        elif self.pooling_mode == "mean":
            combined = torch.mean(x, dim=1)
        elif self.pooling_mode == "attention":
            combined = self.attn_pool(x)

        logits = self.classifier(combined)
        return logits


def get_feature_extractor(device: torch.device) -> nn.Module:
    """
    Loads pretrained EfficientNet-B0, strips classification head, and freezes weights.
    Outputs a (1280,) feature embedding per image frame.
    """
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Extract only convolutional feature extraction + adaptive avgpool
    # backbone.features -> (B, 1280, 7, 7)
    # backbone.avgpool -> (B, 1280, 1, 1)
    # backbone.classifier -> Dropout + Linear (ignored)
    extractor = nn.Sequential(
        backbone.features,
        backbone.avgpool,
        nn.Flatten(),
    )
    extractor.to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor
