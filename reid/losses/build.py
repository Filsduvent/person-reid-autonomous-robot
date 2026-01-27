import torch.nn as nn
from reid.losses.triplet import BatchHardTripletLoss

class LossBundle(nn.Module):
    def __init__(self, triplet=None, w_triplet=1.0):
        super().__init__()
        self.triplet = triplet
        self.w_triplet = float(w_triplet)

    def forward(self, embeddings, labels):
        total = 0.0
        logs = {}
        if self.triplet is not None:
            lt = self.triplet(embeddings, labels)
            total = total + self.w_triplet * lt
            logs["loss/triplet"] = float(lt.detach().cpu())
        logs["loss/total"] = float(total.detach().cpu()) if hasattr(total, "detach") else float(total)
        return total, logs

def build_criterion(cfg):
    lcfg = cfg["loss"]
    trip = None
    w_trip = 1.0

    if lcfg["triplet"]["enabled"]:
        trip = BatchHardTripletLoss(margin=float(lcfg["triplet"]["margin"]))
        w_trip = float(lcfg["triplet"]["weight"])

    return LossBundle(triplet=trip, w_triplet=w_trip)
