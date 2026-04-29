import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from reid.metrics.rerank import re_ranking
from reid.models.baseline import ReidBaseline


def test_re_ranking_returns_query_gallery_matrix():
    q_g_dist = np.array(
        [
            [0.1, 0.7, 0.9],
            [0.8, 0.2, 0.4],
        ],
        dtype=np.float32,
    )
    q_q_dist = np.array(
        [
            [0.0, 0.5],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    g_g_dist = np.array(
        [
            [0.0, 0.6, 0.7],
            [0.6, 0.0, 0.3],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float32,
    )

    out = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=2, k2=1, lambda_value=0.3)

    assert out.shape == q_g_dist.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_re_ranking_random_small_matrices_are_finite():
    rng = np.random.default_rng(0)
    q_g = rng.random((3, 5), dtype=np.float32)
    q_q = rng.random((3, 3), dtype=np.float32)
    g_g = rng.random((5, 5), dtype=np.float32)

    dist = re_ranking(q_g, q_q, g_g)

    assert dist.shape == q_g.shape
    assert np.isfinite(dist).all()


def test_re_ranking_rejects_incompatible_shapes():
    q_g_dist = np.zeros((2, 3), dtype=np.float32)
    q_q_dist = np.zeros((3, 3), dtype=np.float32)
    g_g_dist = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(AssertionError):
        re_ranking(q_g_dist, q_q_dist, g_g_dist)


@pytest.mark.parametrize("rerank_enabled", [False, True])
def test_evaluate_reid_optionally_reports_rerank_metrics(rerank_enabled):
    pytest.importorskip("sklearn")
    from reid.engine.evaluator import evaluate_reid

    samples = [
        (torch.randn(3, 256, 128), 0, 0, "q0.jpg", 0),
        (torch.randn(3, 256, 128), 1, 0, "q1.jpg", 0),
        (torch.randn(3, 256, 128), 0, 1, "g0.jpg", 1),
        (torch.randn(3, 256, 128), 1, 1, "g1.jpg", 1),
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    cfg = {
        "eval": {
            "normalize_feat": True,
            "distance": "euclidean",
            "topk": [1],
            "rerank": {
                "enabled": rerank_enabled,
                "k1": 2,
                "k2": 1,
                "lambda_value": 0.3,
            },
        }
    }
    model = ReidBaseline(
        pretrained=False,
        last_conv_stride=1,
        embedding_dim=128,
        bnneck=True,
        normalize=True,
        metric_feat="raw",
        eval_feat="bn",
        classifier_enabled=False,
        num_classes=None,
    )

    result = evaluate_reid(cfg, model, loader, torch.device("cpu"))

    assert "mAP" in result
    assert "mINP" in result
    assert "Rank1" in result
    if rerank_enabled:
        assert "rerank_mAP" in result
        assert "rerank_mINP" in result
        assert "rerank_Rank1" in result
        assert "rerank_cmc" in result
    else:
        assert "rerank_mAP" not in result
        assert "rerank_mINP" not in result
        assert "rerank_Rank1" not in result
