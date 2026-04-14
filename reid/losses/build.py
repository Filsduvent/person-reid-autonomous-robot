import torch.nn as nn
from reid.losses.triplet import BatchHardTripletLoss
from reid.losses.id import CrossEntropyLabelSmooth

class LossBundle(nn.Module):
    def __init__(self, triplet=None, w_triplet=1.0, id_loss=None, w_id=1.0):
        super().__init__()
        self.triplet = triplet
        self.w_triplet = float(w_triplet)
        self.id_loss = id_loss
        self.w_id = float(w_id)

    def forward(self, embeddings, labels, logits=None):
        total = 0.0
        logs = {}
        if self.triplet is not None:
            lt = self.triplet(embeddings, labels)
            total = total + self.w_triplet * lt
            logs["loss/triplet"] = float(lt.detach().cpu())
        if self.id_loss is not None:
            if logits is None:
                raise ValueError("ID loss enabled but logits are None.")
            li = self.id_loss(logits, labels)
            total = total + self.w_id * li
            logs["loss/id"] = float(li.detach().cpu())
        logs["loss/total"] = float(total.detach().cpu()) if hasattr(total, "detach") else float(total)
        return total, logs

def build_criterion(cfg):
    lcfg = cfg["loss"]
    trip = None
    w_trip = 1.0
    id_loss = None
    w_id = 1.0

    if lcfg["triplet"]["enabled"]:
        trip = BatchHardTripletLoss(margin=float(lcfg["triplet"]["margin"]))
        w_trip = float(lcfg["triplet"]["weight"])

    if "id" in lcfg and lcfg["id"]["enabled"]:
        id_loss = CrossEntropyLabelSmooth(epsilon=float(lcfg["id"]["label_smoothing"]))
        w_id = float(lcfg["id"]["weight"])

    return LossBundle(triplet=trip, w_triplet=w_trip, id_loss=id_loss, w_id=w_id)
