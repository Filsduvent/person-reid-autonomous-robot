import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center loss from Wen et al.

    Args:
        num_classes: Number of identity classes.
        feat_dim: Feature dimensionality used for center loss.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be > 0")
        if int(feat_dim) <= 0:
            raise ValueError("feat_dim must be > 0")

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [B, D], got {tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [B], got {tuple(labels.shape)}")
        if features.size(0) != labels.size(0):
            raise ValueError("Batch size mismatch between features and labels.")
        if features.size(1) != self.feat_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feat_dim}, got {features.size(1)}"
            )

        labels = labels.long()
        centers_batch = self.centers.index_select(0, labels)
        diff = features - centers_batch
        loss = 0.5 * diff.pow(2).sum(dim=1).mean()
        return loss
