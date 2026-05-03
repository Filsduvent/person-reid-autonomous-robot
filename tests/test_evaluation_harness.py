import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reid.engine.evaluator import evaluate_reid, extract_features


BASE_CFG = {
    "eval": {
        "normalize_feat": True,
        "distance": "euclidean",
        "topk": [1],
        "rerank": {
            "enabled": False,
            "k1": 2,
            "k2": 1,
            "lambda_value": 0.3,
        },
    }
}


class EmbOnlyModel(nn.Module):
    def forward(self, x):
        return {"emb": x.float()}


class MissingEmbModel(nn.Module):
    def forward(self, x):
        return {
            "feat_raw": x.float(),
            "feat_bn": x.float(),
            "logits": None,
        }


def _loader():
    samples = [
        (torch.tensor([1.0, 0.0]), 0, 0, "q0.jpg", 0),
        (torch.tensor([0.0, 1.0]), 1, 0, "q1.jpg", 0),
        (torch.tensor([1.0, 0.0]), 0, 1, "g0.jpg", 1),
        (torch.tensor([0.0, 1.0]), 1, 1, "g1.jpg", 1),
    ]
    return DataLoader(samples, batch_size=2, shuffle=False)


@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
def test_evaluator_uses_embedding_only_and_always_reports_core_metrics(distance):
    cfg = {
        **BASE_CFG,
        "eval": {
            **BASE_CFG["eval"],
            "distance": distance,
            "rerank": {
                **BASE_CFG["eval"]["rerank"],
                "enabled": False,
            },
        },
    }

    scores = evaluate_reid(cfg, EmbOnlyModel(), _loader(), torch.device("cpu"))

    for key in ["mAP", "mINP", "Rank1", "Rank5", "Rank10"]:
        assert key in scores
        assert scores[key] is not None
    assert scores["cmc"] == [scores["Rank1"]]
    assert "rerank_mAP" not in scores
    assert "rerank_mINP" not in scores
    assert "rerank_Rank1" not in scores
    assert "rerank_Rank5" not in scores
    assert "rerank_Rank10" not in scores


def test_evaluator_reports_rerank_metrics_separately_without_replacing_original_metrics():
    cfg = {
        **BASE_CFG,
        "eval": {
            **BASE_CFG["eval"],
            "rerank": {
                **BASE_CFG["eval"]["rerank"],
                "enabled": True,
            },
        },
    }

    scores = evaluate_reid(cfg, EmbOnlyModel(), _loader(), torch.device("cpu"))

    for key in ["mAP", "mINP", "Rank1", "Rank5", "Rank10"]:
        assert key in scores
        assert scores[key] is not None
    for key in ["rerank_mAP", "rerank_mINP", "rerank_Rank1", "rerank_Rank5", "rerank_Rank10"]:
        assert key in scores
        assert scores[key] is not None
    assert scores["cmc"] == [scores["Rank1"]]
    assert scores["rerank_cmc"] == [scores["rerank_Rank1"]]


def test_extract_features_rejects_models_without_embedding_output():
    with pytest.raises(ValueError, match="emb"):
        extract_features(MissingEmbModel(), _loader(), torch.device("cpu"))
