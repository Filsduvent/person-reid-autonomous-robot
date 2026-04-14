import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy with label smoothing.
    epsilon=0 reduces to standard cross-entropy.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0:
            return F.cross_entropy(logits, targets)
        n = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            # smoothing across all classes
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / n)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon + (self.epsilon / n))
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
