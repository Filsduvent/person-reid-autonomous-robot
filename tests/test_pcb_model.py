"""PCB backbone and stripe pooling tests, independent of network/cache."""

import hashlib

import pytest
import torch
from torch import nn
from torchvision.models.resnet import Bottleneck

import reid.models.pcb as pcb


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary PCB tests must not download weights")
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)


def historical_fixture(model):
    # Deterministic local raw ImageNet-like state, omitting historical BN counters.
    state = {key: value.clone() for key, value in model.state_dict().items()
             if not key.endswith("num_batches_tracked")}
    state["conv1.weight"].fill_(0.125)
    state["bn1.running_mean"].fill_(0.25)
    state["fc.weight"] = torch.zeros(1000, 2048)
    state["fc.bias"] = torch.zeros(1000)
    return state


def test_backbone_topology_strides_dilation_and_no_heads():
    model = pcb.PCBBackbone(pretrained=False)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    assert [len(stage) for stage in stages] == [3, 4, 6, 3]
    for stage, channels, stride in zip(stages, [256, 512, 1024, 2048], [1, 2, 2, 1]):
        first = stage[0]
        assert isinstance(first, Bottleneck)
        assert first.conv1.stride == first.conv3.stride == (1, 1)
        assert first.conv2.stride == first.downsample[0].stride == (stride, stride)
        assert first.stride == stride
        assert first.conv3.out_channels == channels
        for block in stage:
            assert block.expansion == 4
            assert block.conv2.kernel_size == (3, 3)
            assert block.conv2.padding == block.conv2.dilation == (1, 1)
            assert block.conv2.groups == 1
            for conv in (block.conv1, block.conv2, block.conv3):
                assert conv.bias is None
        for block in stage[1:]:
            assert block.conv2.stride == (1, 1)
            assert block.downsample is None
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert model.maxpool.kernel_size == 3 and model.maxpool.stride == 2
    assert not any(isinstance(m, (nn.Linear, nn.AdaptiveAvgPool2d, nn.Dropout)) for m in model.modules())
    assert not hasattr(model, "fc") and not hasattr(model, "avgpool")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.eps == 1e-5 and m.momentum == 0.1
            assert torch.equal(m.weight, torch.ones_like(m.weight))
            assert torch.equal(m.bias, torch.zeros_like(m.bias))


@pytest.mark.parametrize("training", [False, True])
def test_standard_input_spatial_shape(training):
    model = pcb.PCBBackbone(pretrained=False).train(training)
    with torch.no_grad():
        y = model(torch.randn(1, 3, 384, 128))
    assert y.shape == (1, 2048, 24, 8)
    assert torch.isfinite(y).all()


def test_gradient_flows_through_all_stages_without_optimizer():
    torch.manual_seed(5)
    model = pcb.PCBBackbone(pretrained=False).train()
    x = torch.randn(1, 3, 384, 128, requires_grad=True)
    model(x).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for weight in (model.conv1.weight, model.layer1[0].conv2.weight,
                   model.layer2[0].conv2.weight, model.layer3[0].conv2.weight,
                   model.layer4[0].conv2.weight, model.layer4[0].downsample[0].weight):
        assert torch.count_nonzero(weight.grad) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_pretrained_disabled_never_reads_weights(monkeypatch):
    monkeypatch.setattr(pcb, "_read_historical_weights", lambda *a: pytest.fail("weight read"))
    real_resnet = pcb.resnet50
    def uninitialized_resnet(**kwargs):
        assert kwargs == {"weights": None}
        return real_resnet(**kwargs)
    monkeypatch.setattr(pcb, "resnet50", uninitialized_resnet)
    pcb.PCBBackbone(pretrained=False)
    with pytest.raises(ValueError, match="requires pretrained"):
        pcb.PCBBackbone(pretrained=False, weights_path="unused.pth")


def test_historical_mapping_and_explicit_bn_counter_reset():
    model = pcb.PCBBackbone(pretrained=False)
    source = historical_fixture(model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.num_batches_tracked.fill_(7)
    model.load_imagenet_state_dict(source)
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            assert value.item() == 0
        else:
            assert torch.equal(value, source[key]), key
    assert "fc.weight" in source  # caller state was not mutated
    assert not any(key.endswith("num_batches_tracked") for key in source)


def test_pretrained_constructor_uses_historical_reader(monkeypatch):
    source = historical_fixture(pcb.PCBBackbone(pretrained=False))
    calls = []
    def read(path):
        calls.append(path)
        return source
    monkeypatch.setattr(pcb, "_read_historical_weights", read)
    model = pcb.PCBBackbone(pretrained=True, weights_path="local-reference.pth")
    assert calls == ["local-reference.pth"]
    assert torch.equal(model.conv1.weight, source["conv1.weight"])


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "dtype", "bad_fc", "extra_fc", "bad_counter"])
def test_bad_imagenet_state_rejected_before_mutation(mutation):
    model = pcb.PCBBackbone(pretrained=False)
    original = model.conv1.weight.detach().clone()
    state = historical_fixture(model)
    if mutation == "missing":
        del state["bn1.running_var"]
    elif mutation == "unexpected":
        state["other.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state["layer4.0.conv2.weight"] = torch.zeros(1)
    elif mutation == "dtype":
        state["bn1.weight"] = state["bn1.weight"].double()
    elif mutation == "bad_fc":
        state["fc.weight"] = torch.zeros(3, 2048)
    elif mutation == "extra_fc":
        state["fc.extra"] = torch.zeros(1)
    else:
        state["bn1.num_batches_tracked"] = torch.zeros(1)
    with pytest.raises(ValueError, match="ImageNet"):
        model.load_imagenet_state_dict(state)
    assert torch.equal(model.conv1.weight, original)


def test_reference_source_constants():
    assert pcb.HISTORICAL_IMAGENET_URL == "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    assert pcb.HISTORICAL_IMAGENET_SHA256 == "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"


@pytest.mark.parametrize("cached", [False, True])
def test_download_and_cache_checksum_gate(tmp_path, monkeypatch, cached):
    content = b"test-only mock artifact; never deserialized"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(pcb, "HISTORICAL_IMAGENET_SHA256", digest)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    path = tmp_path / "checkpoints" / "resnet50-19c8e357.pth"
    downloads = []
    def download(url, destination, **kwargs):
        assert url == pcb.HISTORICAL_IMAGENET_URL
        assert kwargs == {"hash_prefix": digest, "progress": False}
        downloads.append(destination)
        path.write_bytes(content)
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    if cached:
        path.parent.mkdir()
        path.write_bytes(content)
    def load(stream, **kwargs):
        assert stream.read() == content
        assert kwargs == {"map_location": "cpu", "weights_only": False}
        return {"mock": "verified"}
    monkeypatch.setattr(torch, "load", load)
    assert pcb._read_historical_weights() == {"mock": "verified"}
    assert len(downloads) == (0 if cached else 1)


@pytest.mark.parametrize("local", [False, True])
def test_wrong_checksum_never_reaches_legacy_deserializer(tmp_path, monkeypatch, local):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    path = tmp_path / "checkpoints" / "resnet50-19c8e357.pth"
    path.parent.mkdir()
    path.write_bytes(b"wrong weights")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unsafe load before verification"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        pcb._read_historical_weights(path if local else None)


def test_missing_local_weights_do_not_substitute():
    with pytest.raises(FileNotFoundError):
        pcb._read_historical_weights("/does-not-exist/reference-weights.pth")


def test_stripe_boundaries_order_full_width_and_exact_coverage():
    rows = torch.arange(24, dtype=torch.float64).view(1, 1, 24, 1)
    feature_map = rows.expand(2, 3, 24, 8).clone()
    stripes = pcb.PCBStripePool.partition(feature_map)
    assert isinstance(stripes, tuple) and len(stripes) == 6
    coverage = torch.zeros(24, dtype=torch.int64)
    for index, stripe in enumerate(stripes):
        assert stripe.shape == (2, 3, 4, 8)
        expected_rows = torch.arange(index * 4, index * 4 + 4, dtype=torch.float64)
        torch.testing.assert_close(stripe, expected_rows.view(1, 1, 4, 1).expand_as(stripe),
                                   rtol=0, atol=0)
        coverage[stripe[0, 0, :, 0].long()] += 1
    assert torch.equal(coverage, torch.ones_like(coverage))
    torch.testing.assert_close(torch.cat(stripes, dim=2), feature_map, rtol=0, atol=0)
    pooled = pcb.PCBStripePool()(feature_map)
    for actual, expected_mean in zip(pooled, (1.5, 5.5, 9.5, 13.5, 17.5, 21.5)):
        torch.testing.assert_close(actual, torch.full((2, 3, 1, 1), expected_mean,
                                                    dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("height,width", [(6, 1), (12, 3), (24, 8), (30, 5)])
def test_stripe_means_preserve_batches_channels_and_average_complete_width(height, width):
    batch = torch.arange(2, dtype=torch.float64).view(2, 1, 1, 1) * 10000
    channel = torch.arange(3, dtype=torch.float64).view(1, 3, 1, 1) * 1000
    row = torch.arange(height, dtype=torch.float64).view(1, 1, height, 1) * 10
    column = torch.arange(width, dtype=torch.float64).view(1, 1, 1, width)
    feature_map = batch + channel + row + column
    before = feature_map.clone()
    pool = pcb.PCBStripePool()
    assert not list(pool.parameters()) and not list(pool.buffers())
    pooled = pool(feature_map)
    stripe_height = height // 6
    for index, actual in enumerate(pooled):
        expected_row_mean = (index * stripe_height + (stripe_height - 1) / 2) * 10
        expected = batch + channel + expected_row_mean + (width - 1) / 2
        assert actual.shape == (2, 3, 1, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(pool.eval()(feature_map), pooled):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(feature_map, before, rtol=0, atol=0)


@pytest.mark.parametrize("height", [0, 1, 5, 7, 16, 25])
def test_stripe_pool_rejects_invalid_height_before_pooling(height, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid heights must fail before any pooling")
    monkeypatch.setattr(pcb.F, "avg_pool2d", forbidden)
    monkeypatch.setattr(pcb.F, "adaptive_avg_pool2d", forbidden)
    with pytest.raises(ValueError, match="height must be positive and divisible by 6"):
        pcb.PCBStripePool()(torch.empty(1, 2, height, 8))


@pytest.mark.parametrize("feature_map,message", [
    (torch.empty(1, 2, 24, 0), "width must be positive"),
    (torch.empty(2, 24, 8), "4D"),
    (None, "4D"),
])
def test_stripe_pool_rejects_invalid_input(feature_map, message):
    with pytest.raises(ValueError, match=message):
        pcb.PCBStripePool()(feature_map)


def test_each_stripe_gradient_is_uniform_inside_and_zero_outside():
    feature_map = torch.randn(2, 3, 24, 8, dtype=torch.float64, requires_grad=True)
    pooled = pcb.PCBStripePool()(feature_map)
    for index, output in enumerate(pooled):
        gradient, = torch.autograd.grad(output.sum(), feature_map, retain_graph=True)
        expected = torch.zeros_like(feature_map)
        expected[:, :, index * 4:(index + 1) * 4, :] = 1 / 32
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    # A distinct coefficient for each part and channel catches mixing/order errors.
    channel_weights = torch.tensor([1., 2., 4.], dtype=torch.float64).view(1, 3, 1, 1)
    loss = sum((index + 1) * (output * channel_weights).sum()
               for index, output in enumerate(pooled))
    loss.backward()
    expected = torch.empty_like(feature_map)
    for index in range(6):
        expected[:, :, index * 4:(index + 1) * 4, :] = (index + 1) * channel_weights / 32
    assert torch.isfinite(feature_map.grad).all()
    torch.testing.assert_close(feature_map.grad, expected, rtol=0, atol=0)


def test_backbone_to_six_pooled_stripes_without_adaptive_pooling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("PCB stripes must use full-stripe pooling, not adaptive pooling")
    monkeypatch.setattr(pcb.F, "adaptive_avg_pool2d", forbidden)
    backbone = pcb.PCBBackbone(pretrained=False).eval()
    pool = pcb.PCBStripePool()
    with torch.no_grad():
        feature_map = backbone(torch.randn(1, 3, 384, 128))
        assert feature_map.shape == (1, 2048, 24, 8)
        stripes = pool.partition(feature_map)
        assert all(stripe.shape == (1, 2048, 4, 8) for stripe in stripes)
        pooled = pool(feature_map)
    assert isinstance(pooled, tuple) and len(pooled) == 6
    for index, actual in enumerate(pooled):
        assert actual.shape == (1, 2048, 1, 1)
        expected = feature_map[:, :, index * 4:(index + 1) * 4, :].mean((2, 3), keepdim=True)
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()
