"""CPU MGN architecture oracles; no network and no production training."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision.models import resnet50

import reid.models.mgn as mgn


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('MGN tests must never download weights')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)


def test_backbone_geometry_split_and_independent_storage():
    model = mgn.MGNBackbone().eval()
    assert len(model.shared.layer1) == 3 and len(model.shared.layer2) == 4
    assert all(len(b.layer3) == 5 and len(b.layer4) == 3 for b in model.branches)
    assert not any(isinstance(m, nn.Linear) for m in model.modules())
    reference = model.branches[0].state_dict()
    for branch, stride in zip(model.branches, [2, 1, 1]):
        first = branch.layer4[0]
        assert first.conv2.stride == first.downsample[0].stride == (stride, stride)
        assert all(m.dilation == (1, 1) for m in branch.modules() if isinstance(m, nn.Conv2d))
        for key, tensor in branch.state_dict().items():
            torch.testing.assert_close(tensor, reference[key], rtol=0, atol=0)
            if branch is not model.branches[0]:
                assert tensor.data_ptr() != reference[key].data_ptr()
    shared_shapes = []
    handle = model.shared.register_forward_hook(lambda mod, args, value: shared_shapes.append(value.shape))
    with torch.no_grad():
        maps = model(torch.randn(1, 3, 384, 128))
    handle.remove()
    assert shared_shapes == [torch.Size([1, 1024, 24, 8])]
    assert [tuple(v.shape) for v in maps] == [(1, 2048, 12, 4), (1, 2048, 24, 8), (1, 2048, 24, 8)]


def test_backbone_disabled_initialization_no_artifact_access(monkeypatch):
    monkeypatch.setattr(mgn, '_read_historical_weights', lambda *a: pytest.fail('artifact read'))
    model = mgn.MGNBackbone()
    with pytest.raises(ValueError, match='requires pretrained'):
        mgn.MGNBackbone(weights_path='unused')
    before = model.shared.bn1.num_batches_tracked.clone()
    with pytest.raises(ValueError, match='384,128'):
        model(torch.randn(2, 3, 256, 128))
    assert torch.equal(model.shared.bn1.num_batches_tracked, before)


def test_historical_mapping_copies_all_residual_values_and_buffers(monkeypatch):
    source = resnet50(weights=None).state_dict()
    source = {k: v for k, v in source.items() if not k.endswith('num_batches_tracked')}
    source['conv1.weight'].fill_(.125)
    source['layer4.0.bn1.running_mean'].fill_(.25)
    monkeypatch.setattr(mgn, '_read_historical_weights', lambda path: source)
    model = mgn.MGNBackbone(pretrained=True, weights_path='fixture')
    for name, value in model.shared.state_dict().items():
        original = name.replace('layer3_0.', 'layer3.0.')
        wanted = torch.zeros_like(value) if name.endswith('num_batches_tracked') else source[original]
        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    for branch in model.branches:
        for name, value in branch.state_dict().items():
            parts = name.split('.')
            if parts[0] == 'layer3':
                parts[1] = str(int(parts[1])+1)
            original = '.'.join(parts)
            wanted = torch.zeros_like(value) if name.endswith('num_batches_tracked') else source[original]
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)


@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'dtype', 'counter'])
def test_historical_mapping_rejects_malformed_state_before_load(damage):
    base = resnet50(weights=None)
    state = dict(base.state_dict())
    if damage == 'missing': del state['layer4.2.conv3.weight']
    elif damage == 'unexpected': state['unknown'] = torch.zeros(1)
    elif damage == 'shape': state['fc.weight'] = torch.zeros(1, 2048)
    elif damage == 'dtype': state['conv1.weight'] = state['conv1.weight'].double()
    else: state['bn1.num_batches_tracked'] = torch.zeros((), dtype=torch.float32)
    with pytest.raises(ValueError, match='ImageNet'):
        mgn._load_imagenet_state(base, state)


def test_artifact_hash_precedes_deserialization_and_missing_fails(tmp_path, monkeypatch):
    wrong = tmp_path/'wrong.pth'
    wrong.write_bytes(b'untrusted')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('deserialization before hash'))
    with pytest.raises(ValueError, match='SHA256'):
        mgn._read_historical_weights(wrong)
    with pytest.raises(FileNotFoundError):
        mgn._read_historical_weights(tmp_path/'missing.pth')


def test_exact_existing_historical_artifact_reader():
    path = Path('/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
    assert path.stat().st_size == 102502400
    assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == mgn.IMAGENET_SHA256
    state = mgn._read_historical_weights(path)
    base = resnet50(weights=None)
    mgn._load_imagenet_state(base, state)
    assert torch.equal(base.conv1.weight, state['conv1.weight'])


def test_max_pooling_global_and_stripe_boundaries_exact_order():
    maps = [torch.arange(h*w, dtype=torch.float32).reshape(1, 1, h, w).expand(2, 2048, h, w).clone()+offset
            for h, w, offset in [(12, 4, 0), (24, 8, 100), (24, 8, 1000)]]
    for feature in maps: feature.requires_grad_()
    pooled = mgn.MGNMaxPool()(maps)
    expected = [47, 291, 1191, 195, 291, 1063, 1127, 1191]
    for value, wanted in zip(pooled, expected):
        assert value.shape == (2, 2048, 1, 1)
        assert torch.equal(value, torch.full_like(value, wanted))
    pooled[3].sum().backward()
    assert maps[1].grad[:, :, :12].count_nonzero() == 2*2048
    assert maps[1].grad[:, :, 12:].count_nonzero() == 0


def test_reductions_template_values_independent_storage_and_initializer(monkeypatch):
    calls = []
    original = nn.init.kaiming_normal_
    def kaiming(tensor, *args, **kwargs):
        calls.append(kwargs)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(nn.init, 'kaiming_normal_', kaiming)
    reductions = mgn.MGNReductions()
    assert calls == [{'mode': 'fan_in'}]
    assert len(reductions.parts) == 8
    for part in reductions.parts:
        assert part[0].bias is None and part[0].kernel_size == (1, 1)
        assert part[1].eps == 1e-5 and part[1].momentum == .1
        assert part[1].affine and part[1].track_running_stats
        assert not part[2].inplace
    states = [part.state_dict() for part in reductions.parts]
    for key in states[0]:
        assert len({state[key].data_ptr() for state in states}) == 8
        for state in states[1:]:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)
    assert torch.equal(states[0]['1.running_mean'], torch.zeros(256))
    assert torch.equal(states[0]['1.running_var'], torch.ones(256))
    reductions.parts[0][1].running_mean.fill_(3)
    assert reductions.parts[1][1].running_mean.count_nonzero() == 0


def test_reduction_numerical_order_and_attached_gradients():
    reductions = mgn.MGNReductions().eval()
    pooled = [torch.randn(2, 2048, 1, 1, requires_grad=True) for _ in range(8)]
    actual = reductions(pooled)
    expected = []
    for part, feature in zip(reductions.parts, pooled):
        conv, bn = part[:2]
        raw = feature.flatten(1) @ conv.weight.flatten(1).T
        expected.append(((raw-bn.running_mean)/(bn.running_var+bn.eps).sqrt()*bn.weight+bn.bias).relu())
    for value, wanted in zip(actual, expected):
        torch.testing.assert_close(value, wanted, atol=2e-6, rtol=2e-5)
        assert value.shape == (2, 256)
    sum((i+1)*value.square().mean() for i, value in enumerate(actual)).backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in pooled)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in reductions.parameters())


@pytest.mark.parametrize('damage', ['count', 'shape', 'batch', 'dtype', 'device', 'single_train'])
def test_reductions_validate_all_inputs_before_bn_mutation(damage):
    reductions = mgn.MGNReductions().train()
    pooled = [torch.randn(2, 2048, 1, 1) for _ in range(8)]
    if damage == 'count': pooled.pop()
    elif damage == 'shape': pooled[-1] = torch.ones(2, 2048, 2, 1)
    elif damage == 'batch': pooled[-1] = torch.ones(3, 2048, 1, 1)
    elif damage == 'dtype': pooled[-1] = pooled[-1].double()
    elif damage == 'device': pooled[-1] = torch.empty(2, 2048, 1, 1, device='meta')
    else: pooled = [part[:1] for part in pooled]
    before = {k: v.clone() for k, v in reductions.named_buffers()}
    with pytest.raises(ValueError): reductions(pooled)
    assert all(torch.equal(value, before[key]) for key, value in reductions.named_buffers())


def test_pooling_rejects_wrong_geometry():
    maps = [torch.zeros(1, 2048, 12, 4), torch.zeros(1, 2048, 23, 8), torch.zeros(1, 2048, 24, 8)]
    with pytest.raises(ValueError, match='pooling'):
        mgn.MGNMaxPool()(maps)


def test_max_pooling_tie_gradient_matches_standard_maxpool():
    maps = [torch.ones(2, 2048, h, w, requires_grad=True) for h, w in [(12, 4), (24, 8), (24, 8)]]
    mgn.MGNMaxPool()(maps)[0].sum().backward()
    expected = torch.zeros_like(maps[0])
    expected[:, :, 0, 0] = 1
    assert torch.equal(maps[0].grad, expected)
