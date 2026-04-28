import pytest
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from reid.models.baseline import ReidBaseline
from reid.models.build import build_model


def test_resnet50_last_stride_shapes():
    x = torch.randn(2, 3, 256, 128)

    model_s1 = ReidBaseline(
        pretrained=False,
        last_conv_stride=1,
        embedding_dim=128,
        bnneck=True,
        normalize=True,
        metric_feat="bn",
        classifier_enabled=False,
        num_classes=None,
    )
    model_s2 = ReidBaseline(
        pretrained=False,
        last_conv_stride=2,
        embedding_dim=128,
        bnneck=True,
        normalize=True,
        metric_feat="bn",
        classifier_enabled=False,
        num_classes=None,
    )

    model_s1.eval()
    model_s2.eval()

    with torch.no_grad():
        feat_map_s1 = model_s1.backbone(x)
        feat_map_s2 = model_s2.backbone(x)
        out1 = model_s1(x)
        out2 = model_s2(x)

    assert "emb" in out1
    assert "emb" in out2
    assert out1["emb"].shape == (2, 128)
    assert out2["emb"].shape == (2, 128)
    assert feat_map_s1.shape[0] == 2
    assert feat_map_s2.shape[0] == 2
    assert feat_map_s1.shape[1] == feat_map_s2.shape[1] == 2048
    assert feat_map_s1.shape[2] > feat_map_s2.shape[2]
    assert feat_map_s1.shape[3] > feat_map_s2.shape[3]


@pytest.mark.parametrize("bnneck", [True, False])
@pytest.mark.parametrize("eval_feat", ["raw", "bn"])
def test_model_forward_returns_expected_bnneck_outputs(bnneck, eval_feat):
    num_classes = 3
    model = ReidBaseline(
        pretrained=False,
        last_conv_stride=1,
        embedding_dim=128,
        bnneck=bnneck,
        normalize=True,
        metric_feat="raw",
        eval_feat=eval_feat,
        classifier_enabled=True,
        num_classes=num_classes,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 256, 128))

    assert "feat_raw" in outputs
    assert "feat_bn" in outputs
    assert "emb" in outputs
    assert "logits" in outputs
    assert outputs["feat_raw"].shape == outputs["feat_bn"].shape
    assert outputs["emb"].shape[0] == 2
    assert outputs["logits"].shape == (2, num_classes)

    expected_eval_feat = outputs["feat_bn"] if eval_feat == "bn" else outputs["feat_raw"]
    expected_emb = F.normalize(expected_eval_feat, p=2, dim=1)
    assert torch.allclose(outputs["emb"], expected_emb, atol=1e-6, rtol=1e-5)


def test_build_model_uses_configured_last_conv_stride():
    base_cfg = {
        "model": {
            "name": "reid_baseline",
            "backbone": {
                "name": "resnet50",
                "pretrained": False,
                "last_conv_stride": 1,
            },
            "head": {
                "embedding_dim": 128,
                "pooling": "gap",
                "bnneck": True,
                "normalize": True,
                "metric_feat": "bn",
                "eval_feat": "bn",
                "classifier": False,
            },
        }
    }

    model_s1 = build_model(base_cfg)
    base_cfg["model"]["backbone"]["last_conv_stride"] = 2
    model_s2 = build_model(base_cfg)

    assert model_s1.backbone[-1][0].conv2.stride == (1, 1)
    assert model_s1.backbone[-1][0].downsample[0].stride == (1, 1)
    assert model_s2.backbone[-1][0].conv2.stride == (2, 2)
    assert model_s2.backbone[-1][0].downsample[0].stride == (2, 2)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("metric_feat", "bad", "Unsupported metric feature"),
        ("eval_feat", "bad", "Unsupported eval feature"),
    ],
)
def test_build_model_rejects_invalid_feature_selection(field, value, expected_message):
    cfg = {
        "model": {
            "name": "reid_baseline",
            "backbone": {
                "name": "resnet50",
                "pretrained": False,
                "last_conv_stride": 1,
            },
            "head": {
                "embedding_dim": 128,
                "pooling": "gap",
                "bnneck": True,
                "normalize": True,
                "metric_feat": "raw",
                "eval_feat": "bn",
                "classifier": False,
            },
        }
    }
    cfg["model"]["head"][field] = value

    with pytest.raises(ValueError, match=expected_message):
        build_model(cfg)


@pytest.mark.parametrize("eval_feat", ["raw", "bn"])
def test_evaluate_reid_runs_with_both_eval_feature_settings_on_cpu(eval_feat):
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
        }
    }

    for last_conv_stride in (1, 2):
        model = ReidBaseline(
            pretrained=False,
            last_conv_stride=last_conv_stride,
            embedding_dim=128,
            bnneck=True,
            normalize=True,
            metric_feat="raw",
            eval_feat=eval_feat,
            classifier_enabled=False,
            num_classes=None,
        )

        result = evaluate_reid(cfg, model, loader, torch.device("cpu"))

        assert "mAP" in result
        assert "mINP" in result
        assert "Rank1" in result
        assert "cmc" in result
        assert len(result["cmc"]) == 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
def test_resnet50_last_stride_forward_on_cuda_if_available():
    x = torch.randn(2, 3, 256, 128, device="cuda")
    model_s1 = ReidBaseline(
        pretrained=False,
        last_conv_stride=1,
        embedding_dim=128,
        bnneck=True,
        normalize=True,
        metric_feat="bn",
        classifier_enabled=False,
        num_classes=None,
    ).to("cuda")
    model_s2 = ReidBaseline(
        pretrained=False,
        last_conv_stride=2,
        embedding_dim=128,
        bnneck=True,
        normalize=True,
        metric_feat="bn",
        classifier_enabled=False,
        num_classes=None,
    ).to("cuda")
    model_s1.eval()
    model_s2.eval()

    with torch.no_grad():
        feat_map_s1 = model_s1.backbone(x)
        feat_map_s2 = model_s2.backbone(x)
        out1 = model_s1(x)
        out2 = model_s2(x)

    assert out1["emb"].shape == (2, 128)
    assert out2["emb"].shape == (2, 128)
    assert feat_map_s1.shape[2] > feat_map_s2.shape[2]
    assert feat_map_s1.shape[3] > feat_map_s2.shape[3]
