import torch
import torch.nn as nn
import torch.nn.functional as F


class IDCrossEntropyLoss(nn.Module):
    """Standard identity classification loss."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets with shape [B], got {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size mismatch between logits and targets.")
        return F.cross_entropy(logits, targets.long())


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy with label smoothing.
    epsilon=0 reduces to standard cross-entropy.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets with shape [B], got {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size mismatch between logits and targets.")

        targets = targets.long()
        if self.epsilon <= 0.0:
            return F.cross_entropy(logits, targets)

        n = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / n)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon + (self.epsilon / n))
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def build_id_loss(label_smoothing: float = 0.0) -> nn.Module:
    """Return standard CE or label-smoothed CE depending on epsilon."""
    if float(label_smoothing) > 0.0:
        return CrossEntropyLabelSmooth(epsilon=float(label_smoothing))
    return IDCrossEntropyLoss()
