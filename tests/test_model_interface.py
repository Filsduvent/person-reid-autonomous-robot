import copy

import pytest
import torch
import torch.nn.functional as F

from reid.models.build import build_model


BASE_CFG = {
    "model": {
        "name": "reid_baseline",
        "backbone": {
            "name": "resnet50",
            "pretrained": False,
            "last_conv_stride": 1,
        },
        "head": {
            "embedding_dim": 64,
            "pooling": "gap",
            "bnneck": True,
            "normalize": True,
            "metric_feat": "raw",
            "eval_feat": "bn",
            "classifier": True,
        },
    }
}


REQUIRED_MODEL_OUTPUT_KEYS = {"feat_raw", "feat_bn", "emb", "logits"}


def _cfg(*, metric_feat="raw", eval_feat="bn", classifier=True):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["model"]["head"]["metric_feat"] = metric_feat
    cfg["model"]["head"]["eval_feat"] = eval_feat
    cfg["model"]["head"]["classifier"] = classifier
    return cfg


def _assert_model_output_contract(outputs, *, batch_size, feat_dim, num_classes, classifier):
    assert isinstance(outputs, dict)
    assert REQUIRED_MODEL_OUTPUT_KEYS.issubset(outputs.keys())

    assert torch.is_tensor(outputs["feat_raw"])
    assert torch.is_tensor(outputs["feat_bn"])
    assert torch.is_tensor(outputs["emb"])
    assert outputs["feat_raw"].shape == (batch_size, feat_dim)
    assert outputs["feat_bn"].shape == (batch_size, feat_dim)
    assert outputs["emb"].shape == (batch_size, feat_dim)

    if classifier:
        assert torch.is_tensor(outputs["logits"])
        assert outputs["logits"].shape == (batch_size, num_classes)
    else:
        assert outputs["logits"] is None


@pytest.mark.parametrize("classifier", [True, False])
def test_resnet50_baseline_satisfies_model_output_contract(classifier):
    batch_size = 2
    feat_dim = 64
    num_classes = 5 if classifier else None
    model = build_model(_cfg(classifier=classifier), num_classes=num_classes)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(batch_size, 3, 256, 128))

    _assert_model_output_contract(
        outputs,
        batch_size=batch_size,
        feat_dim=feat_dim,
        num_classes=num_classes,
        classifier=classifier,
    )


@pytest.mark.parametrize("eval_feat", ["raw", "bn"])
def test_resnet50_baseline_uses_configured_eval_feature_for_embedding(eval_feat):
    cfg = _cfg(metric_feat="raw", eval_feat=eval_feat, classifier=True)
    model = build_model(cfg, num_classes=5)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 256, 128))

    selected = outputs["feat_raw"] if eval_feat == "raw" else outputs["feat_bn"]
    expected_emb = F.normalize(selected, p=2, dim=1)

    assert torch.allclose(outputs["emb"], expected_emb, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("metric_feat", ["raw", "bn"])
def test_resnet50_baseline_preserves_configured_metric_feature_choice(metric_feat):
    cfg = _cfg(metric_feat=metric_feat, eval_feat="bn", classifier=True)
    model = build_model(cfg, num_classes=5)

    assert model.metric_feat == metric_feat
