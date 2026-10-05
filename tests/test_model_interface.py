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


@pytest.mark.parametrize("heads", [1, 2, 3, 6])
@pytest.mark.parametrize("container", [tuple, list])
def test_generic_logits_validation_preserves_order_and_identity(heads, container):
    from reid.models.outputs import ensure_output_dict, validate_logits
    logits = container(torch.randn(2, 5, requires_grad=True) for _ in range(heads))
    assert validate_logits(logits, batch_size=2) is logits
    output = {"emb": torch.randn(2, 9), "feat_raw": None, "feat_bn": None, "logits": logits}
    assert ensure_output_dict(output) is output
    assert output["logits"] is logits
    assert validate_logits(logits[0]) is logits[0]
    assert validate_logits(None) is None


@pytest.mark.parametrize("case", ["empty", "nested", "mixed", "rank", "batch", "classes", "dtype", "device", "integer", "zero_classes", "label_batch"])
def test_generic_logits_reject_malformed_heads(case):
    from reid.models.outputs import validate_logits
    logits = [torch.zeros(2, 3), torch.zeros(2, 3)]
    expected_batch = 2
    if case == "empty": logits = []
    elif case == "nested": logits[1] = [logits[1]]
    elif case == "mixed": logits[1] = None
    elif case == "rank": logits[1] = torch.zeros(2, 3, 1)
    elif case == "batch": logits[1] = torch.zeros(1, 3)
    elif case == "classes": logits[1] = torch.zeros(2, 4)
    elif case == "dtype": logits[1] = logits[1].double()
    elif case == "device": logits[1] = torch.empty(2, 3, device="meta")
    elif case == "integer": logits = torch.ones(2, 3, dtype=torch.long)
    elif case == "zero_classes": logits = torch.zeros(2, 0)
    else: expected_batch = 7
    with pytest.raises(ValueError, match="[Ll]ogits"):
        validate_logits(logits, batch_size=expected_batch)


def test_output_transport_validates_logits_without_averaging():
    from reid.models.outputs import ensure_output_dict
    embedding = torch.randn(2, 8)
    heads = (torch.randn(2, 3), torch.randn(2, 3))
    transported = ensure_output_dict((embedding, heads))
    assert transported["emb"] is embedding and transported["logits"] is heads
    with pytest.raises(ValueError, match="batch size"):
        ensure_output_dict({"emb": embedding, "logits": torch.randn(3, 3)})


def test_baseline_builder_retains_exact_state_and_forward():
    from reid.models.baseline import ReidBaseline
    cfg = _cfg()
    torch.manual_seed(42)
    built = build_model(cfg, num_classes=5).eval()
    torch.manual_seed(42)
    direct = ReidBaseline(pretrained=False, last_conv_stride=1, embedding_dim=64,
                          bnneck=True, normalize=True, metric_feat="raw", eval_feat="bn",
                          classifier_enabled=True, num_classes=5).eval()
    assert built.embedding_dim == built.feat_dim == 64
    for key, value in direct.state_dict().items():
        torch.testing.assert_close(built.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 32)
        expected, actual = direct(images), built(images)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_unknown_builder_dispatch_precedes_baseline_fields():
    with pytest.raises(NotImplementedError, match="unknown"):
        build_model({"model": {"name": "unknown"}}, num_classes=3)
