import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reid.engine.evaluator import evaluate_reid, extract_features
from reid.metrics.ranking import mean_inp


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


def test_minp_uses_last_valid_positive_and_excludes_same_camera_matches():
    # Sorted gallery order is: same-ID/same-camera (excluded), negative,
    # positive, negative, positive.  The retained positives are ranks 2 and 4,
    # therefore INP is 2 / 4 rather than 1 / 2.
    dist = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]).numpy()
    score = mean_inp(
        dist,
        query_ids=torch.tensor([7]).numpy(),
        gallery_ids=torch.tensor([7, 3, 7, 4, 7]).numpy(),
        query_cams=torch.tensor([1]).numpy(),
        gallery_cams=torch.tensor([1, 2, 2, 2, 3]).numpy(),
    )
    assert score == pytest.approx(0.5)


def test_minp_averages_only_valid_queries():
    dist = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).numpy()
    score = mean_inp(
        dist,
        query_ids=torch.tensor([1, 2]).numpy(),
        gallery_ids=torch.tensor([1, 3]).numpy(),
        query_cams=torch.tensor([0, 0]).numpy(),
        gallery_cams=torch.tensor([1, 1]).numpy(),
    )
    assert score == pytest.approx(1.0)


class AsymmetricFlipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.views = []

    def forward(self, images):
        self.views.append(images.clone())
        return {'emb': torch.stack([images[:, 0, 0, 0], images[:, 0, 0, 1]/3], dim=1)}


def flip_loader():
    image = torch.tensor([[[3., 0.]]])
    return DataLoader([(image, 0, 0, 'q', 0), (image, 0, 1, 'g', 1)], batch_size=2)


@pytest.mark.parametrize('policy,calls,expected', [('none', 1, [3., 0.]), ('mean', 2, [1.5, .5])])
def test_flip_policy_forward_count_raw_aggregation_and_single_metadata(policy, calls, expected):
    import numpy as np
    model = AsymmetricFlipModel()
    features, ids, cams, names, marks = extract_features(model, flip_loader(), torch.device('cpu'), flip_aggregation=policy)
    assert len(model.views) == calls
    np.testing.assert_array_equal(features, np.array([expected, expected], dtype=np.float32))
    np.testing.assert_array_equal(ids, [0, 0])
    np.testing.assert_array_equal(cams, [0, 1])
    np.testing.assert_array_equal(names, ['q', 'g'])
    np.testing.assert_array_equal(marks, [0, 1])
    if policy == 'mean':
        torch.testing.assert_close(model.views[1], torch.flip(model.views[0], (3,)), rtol=0, atol=0)


def test_flip_mean_precedes_global_normalization_and_distance(monkeypatch):
    import copy
    import numpy as np
    import reid.engine.evaluator as evaluator
    cfg = copy.deepcopy(BASE_CFG)
    cfg['eval']['flip_aggregation'] = 'mean'
    captured = []
    real = evaluator.compute_dist
    def distance(query, gallery, **kwargs):
        captured.extend([query.copy(), gallery.copy()])
        return real(query, gallery, **kwargs)
    monkeypatch.setattr(evaluator, 'compute_dist', distance)
    scores = evaluate_reid(cfg, AsymmetricFlipModel(), flip_loader(), torch.device('cpu'))
    for feature in captured:
        np.testing.assert_allclose(feature, np.array([[3., 1.]])/np.sqrt(10), rtol=1e-6)
    assert scores['mAP'] == scores['mINP'] == scores['Rank1'] == 1.


def test_missing_flip_policy_and_explicit_none_preserve_exact_default_scores():
    import copy
    cfg = copy.deepcopy(BASE_CFG)
    before = evaluate_reid(cfg, EmbOnlyModel(), _loader(), torch.device('cpu'))
    cfg['eval']['flip_aggregation'] = 'none'
    after = evaluate_reid(cfg, EmbOnlyModel(), _loader(), torch.device('cpu'))
    assert after == before


@pytest.mark.parametrize('policy', ['bad', 'sum', None, True])
def test_invalid_flip_policy_fails_before_model_forward(policy):
    from reid.utils.config import validate_reid_config
    model = AsymmetricFlipModel()
    with pytest.raises(ValueError, match='flip_aggregation'):
        extract_features(model, flip_loader(), torch.device('cpu'), flip_aggregation=policy)
    assert not model.views
    with pytest.raises(ValueError, match='flip_aggregation'):
        validate_reid_config({'eval': {'flip_aggregation': policy}})


@pytest.mark.parametrize('damage', ['shape', 'batch', 'integer', 'rank', 'dtype', 'device', 'missing'])
def test_flip_embedding_contract_rejects_malformed_second_view(damage):
    class Malformed(nn.Module):
        count = 0
        def forward(self, images):
            self.count += 1
            value = torch.ones(len(images), 2)
            if self.count == 2:
                if damage == 'shape': value = torch.ones(len(images), 3)
                elif damage == 'batch': value = value[:1]
                elif damage == 'integer': value = value.long()
                elif damage == 'rank': value = value.unsqueeze(-1)
                elif damage == 'dtype': value = value.double()
                elif damage == 'device': value = value.to('meta')
                else: value = None
            return {'emb': value}
    with pytest.raises(ValueError, match='[Ff]lip|embeddings'):
        extract_features(Malformed(), flip_loader(), torch.device('cpu'), flip_aggregation='mean')


def test_flip_requires_nchw_while_legacy_feature_only_default_still_works():
    with pytest.raises(ValueError, match='NCHW'):
        extract_features(EmbOnlyModel(), _loader(), torch.device('cpu'), flip_aggregation='mean')
    assert extract_features(EmbOnlyModel(), _loader(), torch.device('cpu'))[0].shape == (4, 2)
