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
