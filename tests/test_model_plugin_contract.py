import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reid.engine.evaluator import evaluate_reid
from reid.engine.train_loop import train_one_epoch
from reid.losses.build import build_criterion
from reid.optim import build_optimizer, build_scheduler


CFG = {
    "model": {
        "head": {
            "metric_feat": "raw",
        },
    },
    "loss": {
        "triplet": {
            "enabled": True,
            "margin": 0.3,
            "weight": 1.0,
        },
        "id": {
            "enabled": True,
            "label_smoothing": 0.1,
            "weight": 1.0,
        },
        "center": {
            "enabled": False,
            "weight": 0.0005,
            "lr": 0.5,
        },
    },
    "optim": {
        "name": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "bias_lr_factor": 1.0,
        "weight_decay_bias": 0.0,
        "momentum": 0.9,
    },
    "sched": {
        "name": "warmup_multistep",
        "milestones": [2],
        "gamma": 0.1,
        "warmup_factor": 0.5,
        "warmup_iters": 1,
        "warmup_method": "linear",
    },
    "eval": {
        "normalize_feat": True,
        "distance": "euclidean",
        "topk": [1],
        "rerank": {
            "enabled": False,
        },
    },
}


class DummyReIDModel(nn.Module):
    def __init__(self, in_dim=4, feat_dim=8, num_classes=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.project = nn.Linear(in_dim, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat_raw = self.project(x.float().flatten(1))
        feat_bn = self.bn(feat_raw)
        emb = F.normalize(feat_bn, p=2, dim=1)
        logits = self.classifier(feat_bn)
        return {
            "feat_raw": feat_raw,
            "feat_bn": feat_bn,
            "emb": emb,
            "logits": logits,
        }


def _cfg():
    return copy.deepcopy(CFG)


def test_dummy_plugin_model_works_with_criterion_train_loop_and_evaluator():
    cfg = _cfg()
    model = DummyReIDModel()
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ),
        batch_size=4,
        shuffle=False,
    )
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    avg_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        aux_optimizer=None,
        device=torch.device("cpu"),
        amp=False,
        log_interval=1,
        scheduler=scheduler,
        epoch=1,
    )

    assert avg_loss >= 0.0

    eval_loader = DataLoader(
        [
            (torch.tensor([1.0, 0.0, 0.0, 0.0]), 0, 0, "q0.jpg", 0),
            (torch.tensor([0.0, 1.0, 0.0, 0.0]), 1, 0, "q1.jpg", 0),
            (torch.tensor([1.0, 0.0, 0.0, 0.0]), 0, 1, "g0.jpg", 1),
            (torch.tensor([0.0, 1.0, 0.0, 0.0]), 1, 1, "g1.jpg", 1),
        ],
        batch_size=2,
        shuffle=False,
    )

    scores = evaluate_reid(cfg, model, eval_loader, torch.device("cpu"))

    assert set(scores) >= {"mAP", "mINP", "Rank1", "Rank5", "Rank10"}
    assert scores["Rank1"] is not None
