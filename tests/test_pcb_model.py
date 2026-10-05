"""PCB component and public output tests, independent of network/cache."""

import hashlib
import math

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


def test_reduction_structure_and_all_parameter_buffer_storage_independent():
    reductions = pcb.PCBPartReductions()
    assert len(reductions.local_conv_list) == 6
    tensors = []
    module_ids = []
    for reduction in reductions.local_conv_list:
        assert isinstance(reduction, nn.Sequential)
        assert [type(module) for module in reduction] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
        conv, bn, relu = reduction
        assert (conv.in_channels, conv.out_channels, conv.kernel_size) == (2048, 256, (1, 1))
        assert conv.stride == conv.dilation == (1, 1) and conv.padding == (0, 0)
        assert conv.groups == 1 and conv.bias is not None
        assert bn.num_features == 256 and bn.affine and bn.track_running_stats
        assert bn.eps == 1e-5 and bn.momentum == 0.1 and relu.inplace
        assert all(parameter.requires_grad for parameter in reduction.parameters())
        tensors.extend(reduction.parameters())
        tensors.extend(reduction.buffers())
        module_ids.extend(id(module) for module in reduction.modules())
    assert len(module_ids) == len(set(module_ids))
    assert len(tensors) == 6 * 7  # Conv weight/bias, BN weight/bias and three buffers.
    assert len({id(tensor) for tensor in tensors}) == len(tensors)
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == len(tensors)
    assert not any(isinstance(module, (nn.Linear, nn.Dropout, nn.AvgPool2d,
                                      nn.AdaptiveAvgPool2d)) for module in reductions.modules())


def test_reduction_initializers_override_constructor_defaults(monkeypatch):
    # Poison modern defaults: final values must come from explicit initialization.
    def poison_conv(module):
        with torch.no_grad():
            module.weight.fill_(17)
            module.bias.fill_(17)
    def poison_bn(module):
        with torch.no_grad():
            for value in (module.weight, module.bias, module.running_mean,
                          module.running_var, module.num_batches_tracked):
                value.fill_(17)
    monkeypatch.setattr(nn.Conv2d, "reset_parameters", poison_conv)
    monkeypatch.setattr(nn.BatchNorm2d, "reset_parameters", poison_bn)
    calls = []
    uniform = nn.init.uniform_
    def record_uniform(tensor, a=0., b=1., **kwargs):
        calls.append((id(tensor), a, b))
        return uniform(tensor, a, b, **kwargs)
    monkeypatch.setattr(nn.init, "uniform_", record_uniform)
    torch.manual_seed(42)
    reductions = pcb.PCBPartReductions()
    expected_calls = []
    bound = 1 / math.sqrt(2048)
    # Reproduce only the specified historical draws, with no modern default draws.
    generator = torch.Generator().manual_seed(42)
    for conv, bn, _ in reductions.local_conv_list:
        expected_calls.extend([(id(conv.weight), -bound, bound),
                               (id(conv.bias), -bound, bound), (id(bn.weight), 0, 1)])
        for value, low, high in ((conv.weight, -bound, bound), (conv.bias, -bound, bound),
                                 (bn.weight, 0, 1)):
            expected = torch.empty_like(value).uniform_(low, high, generator=generator)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert torch.equal(bn.bias, torch.zeros_like(bn.bias))
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))
        assert bn.num_batches_tracked.item() == 0
    assert calls == expected_calls


def test_reduction_initialization_does_not_touch_loaded_backbone(monkeypatch):
    source = historical_fixture(pcb.PCBBackbone(pretrained=False))
    monkeypatch.setattr(pcb, "_read_historical_weights", lambda path: source)
    backbone = pcb.PCBBackbone(pretrained=True, weights_path="mock-historical.pth")
    before = {key: value.clone() for key, value in backbone.state_dict().items()}
    reductions = pcb.PCBPartReductions()
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    backbone_storage = {value.untyped_storage().data_ptr() for value in backbone.state_dict().values()}
    assert backbone_storage.isdisjoint(value.untyped_storage().data_ptr()
                                       for value in reductions.state_dict().values())


def test_reduction_mutation_isolated_in_state_and_outputs():
    reductions = pcb.PCBPartReductions().eval()
    parts = tuple(torch.ones(2, 2048, 1, 1) for _ in range(6))
    before = [{key: value.clone() for key, value in module.state_dict().items()}
              for module in reductions.local_conv_list]
    with torch.no_grad():
        outputs_before = reductions(parts)
        for parameter in reductions.local_conv_list[2].parameters():
            parameter.fill_(2)
        for buffer in reductions.local_conv_list[2].buffers():
            buffer.fill_(3)
        outputs_after = reductions(parts)
    for index, module in enumerate(reductions.local_conv_list):
        if index == 2:
            assert not torch.equal(outputs_before[index], outputs_after[index])
            continue
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[index][key], rtol=0, atol=0)
        torch.testing.assert_close(outputs_before[index], outputs_after[index], rtol=0, atol=0)


def test_reduction_part_and_channel_order_with_flatten_only_after_relu():
    reductions = pcb.PCBPartReductions().double().eval()
    parts = []
    with torch.no_grad():
        for index, (conv, bn, _) in enumerate(reductions.local_conv_list):
            conv.weight.zero_()
            channels = torch.arange(256)
            conv.weight[channels, channels, 0, 0] = index + 1
            conv.bias.zero_()
            bn.weight.fill_(1)
            bn.bias.zero_()
            part = torch.zeros(2, 2048, 1, 1, dtype=torch.float64)
            part[:, :256, 0, 0] = torch.arange(-128, 128) + index * 10
            parts.append(part)
        outputs = reductions(parts)
    assert isinstance(outputs, tuple) and len(outputs) == 6
    for index, output in enumerate(outputs):
        expected = ((torch.arange(-128, 128, dtype=torch.float64) + index * 10)
                    * (index + 1) / (1 + 1e-5) ** 0.5).clamp_min(0)
        assert output.shape == (2, 256)
        torch.testing.assert_close(output, expected.expand(2, 256))


def test_reduction_bn_batch_statistics_running_updates_and_single_item_eval():
    reductions = pcb.PCBPartReductions().double().train()
    parts = []
    with torch.no_grad():
        for index, (conv, bn, _) in enumerate(reductions.local_conv_list):
            conv.weight.zero_()
            conv.weight[:, 0, 0, 0] = index + 1
            conv.bias.fill_(index)
            bn.weight.fill_(0.5)
            bn.bias.fill_(0.25)
            part = torch.zeros(4, 2048, 1, 1, dtype=torch.float64)
            part[:, 0, 0, 0] = torch.tensor([-3., -1., 1., 3.]) + index
            parts.append(part)
        outputs = reductions(parts)
        for index, (conv, bn, _) in enumerate(reductions.local_conv_list):
            values = parts[index][:, 0, 0, 0] * (index + 1) + index
            mean, variance = values.mean(), values.var(unbiased=False)
            expected = ((values - mean) / (variance + bn.eps).sqrt() * 0.5 + 0.25).clamp_min(0)
            assert outputs[index].shape == (4, 256)
            torch.testing.assert_close(outputs[index], expected[:, None].expand(4, 256))
            torch.testing.assert_close(bn.running_mean, (0.1 * mean).expand(256))
            torch.testing.assert_close(bn.running_var, (0.9 + 0.1 * values.var(unbiased=True)).expand(256))
            assert bn.num_batches_tracked.item() == 1
        state = {key: value.clone() for key, value in reductions.state_dict().items()}
        reductions.eval()
        full_eval = reductions(parts)
        single_eval = reductions(tuple(part[:1] for part in parts))
        for index, (_, bn, _) in enumerate(reductions.local_conv_list):
            values = parts[index][:, 0, 0, 0] * (index + 1) + index
            expected = ((values[:, None] - bn.running_mean) /
                        (bn.running_var + bn.eps).sqrt() * bn.weight + bn.bias).clamp_min(0)
            assert full_eval[index].shape == outputs[index].shape
            assert single_eval[index].shape == (1, 256)
            torch.testing.assert_close(full_eval[index], expected)
            torch.testing.assert_close(single_eval[index], expected[:1])
        for key, value in reductions.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_reduction_single_item_training_keeps_standard_bn_error():
    reductions = pcb.PCBPartReductions().train()
    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        reductions(tuple(torch.ones(1, 2048, 1, 1) for _ in range(6)))


@pytest.mark.parametrize("training", [False, True])
def test_gradients_reach_every_reduction_parameter_and_pooled_input(training):
    torch.manual_seed(7)
    reductions = pcb.PCBPartReductions().train(training)
    parts = tuple(torch.randn(4, 2048, 1, 1, requires_grad=True) for _ in range(6))
    outputs = reductions(parts)
    assert all(output.shape == (4, 256) for output in outputs)
    sum((index + 1) * output.square().mean() for index, output in enumerate(outputs)).backward()
    for name, parameter in reductions.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for part in parts:
        assert part.grad is not None and torch.isfinite(part.grad).all()
        assert torch.count_nonzero(part.grad) > 0


def test_single_image_backbone_pool_reduction_forward_and_backward():
    torch.manual_seed(7)
    backbone = pcb.PCBBackbone(pretrained=False).eval()
    reductions = pcb.PCBPartReductions().eval()
    image = torch.randn(1, 3, 384, 128, requires_grad=True)
    feature_map = backbone(image)
    assert feature_map.shape == (1, 2048, 24, 8)
    pooled = pcb.PCBStripePool()(feature_map)
    assert all(part.shape == (1, 2048, 1, 1) for part in pooled)
    outputs = reductions(pooled)
    assert len(outputs) == 6 and all(output.shape == (1, 256) for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.square().mean() for output in outputs).backward()
    for model in (backbone, reductions):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for gradient in (image.grad, backbone.conv1.weight.grad, backbone.layer4[0].conv2.weight.grad):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("malformation", ["count", "tensor", "rank", "channels", "spatial", "batch"])
def test_reduction_rejects_malformed_parts_before_bn_updates(malformation):
    reductions = pcb.PCBPartReductions().train()
    parts = [torch.zeros(2, 2048, 1, 1) for _ in range(6)]
    if malformation == "count":
        parts.pop()
    elif malformation == "tensor":
        parts = torch.zeros(6, 2, 2048, 1, 1)
    elif malformation == "rank":
        parts[-1] = torch.zeros(2, 2048)
    elif malformation == "channels":
        parts[-1] = torch.zeros(2, 256, 1, 1)
    elif malformation == "spatial":
        parts[-1] = torch.zeros(2, 2048, 2, 1)
    else:
        parts[-1] = torch.zeros(3, 2048, 1, 1)
    with pytest.raises(ValueError, match="PCB"):
        reductions(parts)
    assert all(module[1].num_batches_tracked.item() == 0 for module in reductions.local_conv_list)


@pytest.mark.parametrize("num_classes", [1, 3, 17])
def test_identity_classifiers_dynamic_source_classes_and_independence(num_classes):
    heads = pcb.PCBIdentityClassifiers(num_classes)
    assert heads.num_classes == num_classes and len(heads.fc_list) == 6
    assert len({id(head) for head in heads.fc_list}) == 6
    parameters = list(heads.parameters())
    assert len(parameters) == 12 and all(p.requires_grad for p in parameters)
    assert len({id(p) for p in parameters}) == 12
    assert len({p.untyped_storage().data_ptr() for p in parameters}) == 12
    assert not list(heads.buffers())
    for head in heads.fc_list:
        assert type(head) is nn.Linear
        assert head.in_features == 256 and head.out_features == num_classes
        assert head.weight.shape == (num_classes, 256) and head.bias.shape == (num_classes,)
    features = tuple(torch.randn(2, 256) for _ in range(6))
    before = tuple(feature.clone() for feature in features)
    train_logits = heads.train()(features)
    eval_logits = heads.eval()(features)
    assert isinstance(train_logits, tuple) and len(train_logits) == 6
    for feature, saved, train, evaluation in zip(features, before, train_logits, eval_logits):
        assert train.shape == evaluation.shape == (2, num_classes)
        torch.testing.assert_close(train, evaluation, rtol=0, atol=0)
        torch.testing.assert_close(feature, saved, rtol=0, atol=0)


@pytest.mark.parametrize("num_classes", [0, -1, None, True, False, 3.0, 2.5, "3"])
def test_identity_classifiers_reject_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        pcb.PCBIdentityClassifiers(num_classes)


def test_identity_classifier_initialization_is_explicit_and_deterministic(monkeypatch):
    def poison_defaults(module):
        with torch.no_grad():
            module.weight.fill_(17)
            module.bias.fill_(17)
    monkeypatch.setattr(nn.Linear, "reset_parameters", poison_defaults)
    calls = []
    normal = nn.init.normal_
    def record_normal(tensor, mean=0., std=1., **kwargs):
        calls.append((id(tensor), mean, std))
        return normal(tensor, mean, std, **kwargs)
    monkeypatch.setattr(nn.init, "normal_", record_normal)
    torch.manual_seed(42)
    heads = pcb.PCBIdentityClassifiers(7)
    generator = torch.Generator().manual_seed(42)
    for head in heads.fc_list:
        expected = torch.empty_like(head.weight).normal_(0, 0.001, generator=generator)
        torch.testing.assert_close(head.weight, expected, rtol=0, atol=0)
        assert torch.equal(head.bias, torch.zeros_like(head.bias))
    assert calls == [(id(head.weight), 0, 0.001) for head in heads.fc_list]


def test_identity_classifier_initialization_preserves_backbone_and_reductions(monkeypatch):
    source = historical_fixture(pcb.PCBBackbone(pretrained=False))
    monkeypatch.setattr(pcb, "_read_historical_weights", lambda path: source)
    backbone = pcb.PCBBackbone(pretrained=True, weights_path="mock-historical.pth")
    reductions = pcb.PCBPartReductions()
    snapshots = [{key: value.clone() for key, value in model.state_dict().items()}
                 for model in (backbone, reductions)]
    heads = pcb.PCBIdentityClassifiers(3)
    head_storage = {p.untyped_storage().data_ptr() for p in heads.parameters()}
    for model, snapshot in zip((backbone, reductions), snapshots):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
            assert value.untyped_storage().data_ptr() not in head_storage


def test_identity_classifier_mutation_isolation_and_ordered_association():
    heads = pcb.PCBIdentityClassifiers(3).double()
    features = []
    with torch.no_grad():
        for index, head in enumerate(heads.fc_list):
            head.weight.zero_()
            head.weight[:, 0] = torch.tensor([1., 2., 3.]) * (index + 1)
            head.bias.copy_(torch.tensor([10., 20., 30.]) + index)
            feature = torch.zeros(2, 256, dtype=torch.float64)
            feature[:, 0] = torch.tensor([index + 2., index + 12.])
            features.append(feature)
        logits = heads(features)
        for index, actual in enumerate(logits):
            expected = (features[index][:, :1] * (index + 1) * torch.tensor([1., 2., 3.])
                        + torch.tensor([10., 20., 30.]) + index)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        before = {key: value.clone() for key, value in heads.state_dict().items()}
        heads.fc_list[2].weight.add_(5)
        heads.fc_list[2].bias.add_(7)
        changed_logits = heads(features)
        for index, (original, changed) in enumerate(zip(logits, changed_logits)):
            if index == 2:
                assert not torch.equal(original, changed)
            else:
                torch.testing.assert_close(original, changed, rtol=0, atol=0)
                for name, value in heads.fc_list[index].state_dict().items():
                    torch.testing.assert_close(value, before[f"fc_list.{index}.{name}"], rtol=0, atol=0)


@pytest.mark.parametrize("training,batch_size", [(True, 2), (False, 1)])
def test_manual_six_head_ce_through_actual_backbone_reductions_and_classifiers(training, batch_size):
    torch.manual_seed(8)
    # Test-only composition; no public model registration or production CE added.
    components = nn.ModuleDict({
        "backbone": pcb.PCBBackbone(pretrained=False),
        "pool": pcb.PCBStripePool(),
        "reductions": pcb.PCBPartReductions(),
        "classifiers": pcb.PCBIdentityClassifiers(5),
    }).train(training)
    images = torch.randn(batch_size, 3, 384, 128, requires_grad=True)
    feature_map = components["backbone"](images)
    assert feature_map.shape == (batch_size, 2048, 24, 8)
    pooled = components["pool"](feature_map)
    assert all(part.shape == (batch_size, 2048, 1, 1) for part in pooled)
    local_features = components["reductions"](pooled)
    logits = components["classifiers"](local_features)
    assert isinstance(local_features, tuple) and len(local_features) == 6
    assert isinstance(logits, tuple) and len(logits) == 6
    for feature, scores in zip(local_features, logits):
        assert feature.shape == (batch_size, 256) and scores.shape == (batch_size, 5)
        assert torch.isfinite(feature).all() and torch.isfinite(scores).all()
        feature.retain_grad()
    labels = torch.arange(batch_size) % 5
    loss = sum(torch.nn.functional.cross_entropy(scores, labels) for scores in logits)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in components.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for head in components["classifiers"].fc_list:
        assert torch.count_nonzero(head.weight.grad) > 0
        assert torch.count_nonzero(head.bias.grad) > 0
    for reduction, feature in zip(components["reductions"].local_conv_list, local_features):
        assert torch.count_nonzero(reduction[0].weight.grad) > 0
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad) > 0
        assert reduction[1].num_batches_tracked.item() == int(training)
    for gradient in (images.grad, components["backbone"].conv1.weight.grad,
                     components["backbone"].layer4[0].conv2.weight.grad):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("malformation", ["count", "tensor", "rank", "width", "batch", "empty", "none"])
def test_identity_classifiers_reject_malformed_features(malformation):
    heads = pcb.PCBIdentityClassifiers(3)
    features = [torch.zeros(2, 256) for _ in range(6)]
    if malformation == "count":
        features.pop()
    elif malformation == "tensor":
        features = torch.zeros(6, 2, 256)
    elif malformation == "rank":
        features[-1] = torch.zeros(2, 256, 1, 1)
    elif malformation == "width":
        features[-1] = torch.zeros(2, 128)
    elif malformation == "batch":
        features[-1] = torch.zeros(3, 256)
    elif malformation == "empty":
        features[-1] = torch.zeros(0, 256)
    else:
        features[-1] = None
    with pytest.raises(ValueError, match="PCB"):
        heads(features)


class _SentinelLocalFeatures(nn.Module):
    def forward(self, values):
        return tuple(values[:, index:index + 1].expand(-1, 256) for index in range(6))


def _sentinel_pcb():
    # Exercise the real public forward and real classifiers with exact local values.
    model = pcb.PCB(num_classes=3, pretrained=False)
    model.backbone = nn.Identity()
    model.pool = nn.Identity()
    model.reductions = _SentinelLocalFeatures()
    return model


@pytest.mark.parametrize("training", [False, True])
def test_pcb_public_descriptor_exact_order_no_normalization_and_classifier_independence(training):
    from reid.models.outputs import ensure_output_dict, get_embedding

    model = _sentinel_pcb().train(training)
    values = torch.tensor([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]])
    with torch.no_grad():
        outputs = model(values)
        assert set(outputs) == {"emb", "feat_raw", "feat_bn", "logits"}
        assert outputs["feat_raw"] is outputs["feat_bn"] is None
        assert model.embedding_dim == 1536 and model.feat_dim is None
        assert ensure_output_dict(outputs) is outputs
        assert get_embedding(outputs) is outputs["emb"]
        expected = values.repeat_interleave(256, dim=1)
        torch.testing.assert_close(outputs["emb"], expected, rtol=0, atol=0)
        assert outputs["emb"].shape == (2, 1536)
        assert (outputs["emb"].norm(dim=1) > 1).all()
        for index, head in enumerate(model.classifiers.fc_list):
            block = expected[:, index * 256:(index + 1) * 256]
            torch.testing.assert_close(outputs["logits"][index], torch.nn.functional.linear(
                block, head.weight, head.bias))
            head.weight.zero_()
            head.bias.fill_(100 + index)
        changed = model(values)
        torch.testing.assert_close(changed["emb"], outputs["emb"], rtol=0, atol=0)
        for index, logits in enumerate(changed["logits"]):
            assert not torch.equal(logits, outputs["logits"][index])
            torch.testing.assert_close(logits, torch.full((2, 3), 100. + index), rtol=0, atol=0)
        zeros = model(torch.zeros(1, 6))
        assert torch.equal(zeros["emb"], torch.zeros(1, 1536))
        assert torch.isfinite(zeros["emb"]).all()


@pytest.mark.parametrize("training,batch_size", [(True, 2), (False, 1)])
def test_pcb_public_real_forward_and_descriptor_gradient(training, batch_size):
    torch.manual_seed(9)
    model = pcb.PCB(num_classes=7, pretrained=False).train(training)
    assert set(dict(model.named_children())) == {"backbone", "pool", "reductions", "classifiers"}
    assert not any(isinstance(module, (nn.Dropout, nn.AdaptiveAvgPool2d)) for module in model.modules())
    assert all(module.training == training for module in model.modules())
    captured = {}
    def capture(name):
        def hook(module, args, output):
            captured[name] = output
            if training:
                for tensor in output if isinstance(output, tuple) else (output,):
                    tensor.retain_grad()
        return hook
    handles = [module.register_forward_hook(capture(name)) for name, module in
               (("map", model.backbone), ("pooled", model.pool), ("local", model.reductions))]
    images = torch.randn(batch_size, 3, 384, 128, requires_grad=training)
    try:
        with torch.set_grad_enabled(training):
            outputs = model(images)
    finally:
        for handle in handles:
            handle.remove()
    assert set(outputs) == {"emb", "feat_raw", "feat_bn", "logits"}
    assert model.embedding_dim == 1536 and model.feat_dim is None
    assert outputs["feat_raw"] is outputs["feat_bn"] is None
    assert outputs["emb"].shape == (batch_size, 1536)
    assert isinstance(outputs["logits"], tuple) and len(outputs["logits"]) == 6
    assert all(logit.shape == (batch_size, 7) and torch.isfinite(logit).all()
               for logit in outputs["logits"])
    assert torch.isfinite(outputs["emb"]).all()
    torch.testing.assert_close(outputs["emb"], torch.cat(captured["local"], dim=1), rtol=0, atol=0)
    for index, feature in enumerate(captured["local"]):
        assert (feature >= 0).all()  # post-ReLU local representation
        torch.testing.assert_close(outputs["logits"][index], model.classifiers.fc_list[index](feature))
    if training:
        outputs["emb"].square().mean().backward()
        for component in (model.backbone, model.reductions):
            for name, parameter in component.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        for tensor in (captured["map"], *captured["pooled"], *captured["local"], images):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            assert torch.count_nonzero(tensor.grad) > 0
        for reduction in model.reductions.local_conv_list:
            assert torch.count_nonzero(reduction[0].weight.grad) > 0
        assert torch.count_nonzero(model.backbone.conv1.weight.grad) > 0
        # Retrieval branches before classifiers: descriptor loss must not update heads.
        assert all(parameter.grad is None for parameter in model.classifiers.parameters())


def test_pcb_composition_preserves_historical_backbone_and_source_classes(monkeypatch):
    source = historical_fixture(pcb.PCBBackbone(pretrained=False))
    calls = []
    def read(path):
        calls.append(path)
        return source
    monkeypatch.setattr(pcb, "_read_historical_weights", read)
    model = pcb.PCB(num_classes=11, pretrained=True, weights_path="mock-historical.pth")
    assert calls == ["mock-historical.pth"]
    for key, value in model.backbone.state_dict().items():
        expected = torch.zeros_like(value) if key.endswith("num_batches_tracked") else source[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert model.num_classes == 11
    assert all(head.out_features == 11 for head in model.classifiers.fc_list)
    with pytest.raises(ValueError, match="positive integer"):
        pcb.PCB(num_classes=0, pretrained=True)
    assert calls == ["mock-historical.pth"]  # invalid construction must not read weights


def test_actual_pcb_feature_extraction_returns_raw_embedding():
    import numpy as np
    from reid.engine.evaluator import extract_features
    from torch.utils.data import DataLoader

    torch.manual_seed(10)
    model = pcb.PCB(num_classes=3, pretrained=False).eval()
    image = torch.randn(3, 384, 128)
    with torch.no_grad():
        raw = model(image.unsqueeze(0))["emb"].numpy()
    loader = DataLoader([(image, 9, 2, "synthetic.jpg", 0)], batch_size=1)
    model.train()  # extractor owns eval/no-grad behavior
    features, pids, cameras, names, marks = extract_features(model, loader, torch.device("cpu"))
    assert not model.training and features.shape == (1, 1536)
    np.testing.assert_array_equal(features, raw)
    assert pids.tolist() == [9] and cameras.tolist() == [2]
    assert names.tolist() == ["synthetic.jpg"] and marks.tolist() == [0]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("normalize_feat", [False, True])
def test_pcb_evaluator_owns_one_global_normalization(normalize_feat, monkeypatch):
    import numpy as np
    import reid.engine.evaluator as evaluator
    from torch.utils.data import DataLoader

    model = _sentinel_pcb().eval()
    rows = torch.tensor([[1., 2., 3., 4., 5., 6.], [6., 5., 4., 3., 2., 1.],
                         [2., 4., 6., 8., 10., 12.], [12., 10., 8., 6., 4., 2.]])
    samples = [(row, index % 2, int(index >= 2), f"fixture-{index}", int(index >= 2))
               for index, row in enumerate(rows)]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    raw = rows.repeat_interleave(256, dim=1).numpy()
    collected = evaluator.extract_features(model, loader, torch.device("cpu"))[0]
    np.testing.assert_array_equal(collected, raw)
    calls = []
    original_normalize, original_dist = evaluator.normalize, evaluator.compute_dist
    def normalize(values, axis=1):
        calls.append("normalize")
        assert axis == 1
        np.testing.assert_array_equal(values, raw)
        return original_normalize(values, axis=axis)
    def distance(query, gallery, metric):
        calls.append("distance")
        assert metric == "euclidean"
        expected = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12) if normalize_feat else raw
        np.testing.assert_allclose(query, expected[:2], rtol=0, atol=0)
        np.testing.assert_allclose(gallery, expected[2:], rtol=0, atol=0)
        if normalize_feat:
            np.testing.assert_allclose(np.linalg.norm(query, axis=1), np.ones(2), rtol=1e-6)
        return original_dist(query, gallery, metric=metric)
    monkeypatch.setattr(evaluator, "normalize", normalize)
    monkeypatch.setattr(evaluator, "compute_dist", distance)
    cfg = {"eval": {"normalize_feat": normalize_feat, "distance": "euclidean",
                    "topk": [1, 5, 10], "rerank": {"enabled": False}}}
    scores = evaluator.evaluate_reid(cfg, model, loader, torch.device("cpu"))
    assert calls == (["normalize", "distance"] if normalize_feat else ["distance"])
    assert all(np.isfinite(scores[key]) for key in ("mAP", "mINP", "Rank1", "Rank5", "Rank10"))


def test_generic_builder_pcb_output_and_evaluator_without_baseline_fields(monkeypatch):
    from reid.models.build import build_model
    from reid.models.outputs import ensure_output_dict
    from reid.engine.evaluator import extract_features
    from reid.losses.build import build_criterion
    from reid.utils.config import validate_model_loss_requirements, validate_reid_config
    from torch.utils.data import DataLoader
    monkeypatch.setattr(pcb, "_read_historical_weights", lambda *a: pytest.fail("pretraining"))
    cfg = {"model": {"name": "pcb", "pretrained": False},
           "loss": {"id": {"enabled": True, "weight": 1., "label_smoothing": 0.},
                    "triplet": {"enabled": False}, "center": {"enabled": False}}}
    validate_reid_config(cfg, num_classes=7)
    model = build_model(cfg, num_classes=7).eval()
    assert type(model) is pcb.PCB and model.num_classes == 7
    assert model.embedding_dim == 1536 and model.feat_dim is None
    assert all(head.out_features == 7 for head in model.classifiers.fc_list)
    assert model.checkpoint_metadata == {"schema_version": 1, "output_contract_version": 1,
        "model_name": "pcb", "variant": "independent_part_reduction", "num_classes": 7, "embedding_dim": 1536}
    assert validate_model_loss_requirements(cfg, model) is None
    criterion = build_criterion(cfg, num_classes=7, feat_dim=None)
    assert criterion.id_loss is not None and criterion.triplet is criterion.center_loss is None
    image = torch.randn(3, 384, 128)
    with torch.no_grad():
        output = model(image[None])
    assert ensure_output_dict(output) is output
    assert output["emb"].shape == (1, 1536) and output["feat_raw"] is output["feat_bn"] is None
    assert isinstance(output["logits"], tuple) and len(output["logits"]) == 6
    loader = DataLoader([(image, 1, 2, "fixture", 0)], batch_size=1)
    extracted = extract_features(model, loader, torch.device("cpu"))[0]
    torch.testing.assert_close(torch.from_numpy(extracted), output["emb"], rtol=0, atol=0)


def test_generic_pcb_builder_pretraining_and_override(monkeypatch):
    from reid.models.build import build_model
    source = historical_fixture(pcb.PCBBackbone(pretrained=False))
    calls = []
    def read(path):
        calls.append(path)
        return source
    monkeypatch.setattr(pcb, "_read_historical_weights", read)
    cfg = {"model": {"name": "pcb", "weights_path": "historical.pth"}}
    initialized = build_model(cfg, num_classes=3)
    assert calls == ["historical.pth"]  # default new-model initialization is historical
    torch.testing.assert_close(initialized.backbone.conv1.weight, source["conv1.weight"], rtol=0, atol=0)
    monkeypatch.setattr(pcb, "_read_historical_weights", lambda *a: pytest.fail("reconstruction pretraining"))
    restored_shell = build_model(cfg, num_classes=5, initialize_pretrained=False)
    assert restored_shell.classifiers.fc_list[0].out_features == 5
