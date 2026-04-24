import torch
import torch.nn as nn

from reid.losses.center import CenterLoss
from reid.losses.id import build_id_loss
from reid.losses.triplet import BatchHardTripletLoss


class LossBundle(nn.Module):
    def __init__(
        self,
        triplet: nn.Module | None = None,
        w_triplet: float = 1.0,
        id_loss: nn.Module | None = None,
        w_id: float = 1.0,
        center: nn.Module | None = None,
        w_center: float = 1.0,
        center_feat_key: str = "feat_raw",
    ):
        super().__init__()
        self.triplet = triplet
        self.w_triplet = float(w_triplet)
        self.id_loss = id_loss
        self.w_id = float(w_id)
        self.center = center
        self.w_center = float(w_center)
        self.center_feat_key = center_feat_key

    def forward(self, outputs, labels: torch.Tensor):
        if not isinstance(outputs, dict):
            raise TypeError("LossBundle expects model outputs as a dict.")

        labels = labels.long()
        device = labels.device
        total = torch.zeros((), device=device)
        logs = {
            "loss/total": 0.0,
            "loss/triplet": 0.0,
            "loss/id": 0.0,
            "loss/center": 0.0,
        }

        if self.triplet is not None:
            emb = outputs.get("emb")
            if emb is None:
                raise ValueError("Triplet loss enabled but model output 'emb' is missing.")
            lt = self.triplet(emb, labels)
            total = total + self.w_triplet * lt
            logs["loss/triplet"] = float(lt.detach().cpu())

        if self.id_loss is not None:
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("ID loss enabled but model output 'logits' is missing.")
            li = self.id_loss(logits, labels)
            total = total + self.w_id * li
            logs["loss/id"] = float(li.detach().cpu())

        if self.center is not None:
            feat = outputs.get(self.center_feat_key)
            if feat is None:
                raise ValueError(
                    f"Center loss enabled but model output '{self.center_feat_key}' is missing."
                )
            lc = self.center(feat, labels)
            total = total + self.w_center * lc
            logs["loss/center"] = float(lc.detach().cpu())

        logs["loss/total"] = float(total.detach().cpu())
        return total, logs


def build_criterion(cfg, num_classes: int | None, feat_dim: int | None):
    lcfg = cfg["loss"]

    trip = None
    w_trip = 1.0
    if "triplet" in lcfg and lcfg["triplet"]["enabled"]:
        trip = BatchHardTripletLoss(margin=float(lcfg["triplet"]["margin"]))
        w_trip = float(lcfg["triplet"]["weight"])

    id_loss = None
    w_id = 1.0
    if "id" in lcfg and lcfg["id"]["enabled"]:
        id_loss = build_id_loss(label_smoothing=float(lcfg["id"].get("label_smoothing", 0.0)))
        w_id = float(lcfg["id"]["weight"])

    center = None
    w_center = 1.0
    center_feat_key = "feat_raw"
    if "center" in lcfg and lcfg["center"]["enabled"]:
        if num_classes is None or int(num_classes) <= 0:
            raise ValueError("Center loss enabled but num_classes is not set.")
        if feat_dim is None or int(feat_dim) <= 0:
            raise ValueError("Center loss enabled but feat_dim is not set.")
        center = CenterLoss(num_classes=int(num_classes), feat_dim=int(feat_dim))
        w_center = float(lcfg["center"]["weight"])
        center_feat = str(lcfg["center"].get("feat", "raw")).lower()
        if center_feat not in {"raw", "bn"}:
            raise ValueError(f"Unsupported center feature '{center_feat}'. Use 'raw' or 'bn'.")
        center_feat_key = "feat_raw" if center_feat == "raw" else "feat_bn"

    if trip is None and id_loss is None and center is None:
        raise ValueError("At least one loss must be enabled in cfg['loss'].")

    return LossBundle(
        triplet=trip,
        w_triplet=w_trip,
        id_loss=id_loss,
        w_id=w_id,
        center=center,
        w_center=w_center,
        center_feat_key=center_feat_key,
    )
