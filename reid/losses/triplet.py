import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    # x: [N, D]
    # dist(i,j) = ||xi-xj||
    xx = (x * x).sum(dim=1, keepdim=True)              # [N,1]
    dist2 = xx + xx.t() - 2.0 * (x @ x.t())
    dist2 = torch.clamp(dist2, min=1e-12)
    return torch.sqrt(dist2)

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = float(margin)
        self.ranking = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings [N,D], labels [N]
        dist = pairwise_dist(embeddings)
        N = dist.size(0)

        labels = labels.view(-1, 1)
        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos

        # hardest positive: max dist among positives (excluding self is ok because dist=0)
        dist_ap = dist.masked_fill(~is_pos, -1e9).max(dim=1).values
        # hardest negative: min dist among negatives
        dist_an = dist.masked_fill(~is_neg, 1e9).min(dim=1).values

        y = torch.ones_like(dist_an)
        loss = self.ranking(dist_an, dist_ap, y)
        return loss
