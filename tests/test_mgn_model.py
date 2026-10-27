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


@pytest.mark.parametrize('classes', [751, 702, 767, 1041])
def test_classifiers_source_dimensions_bias_and_independent_storage(classes):
    classifiers = mgn.MGNClassifiers(classes)
    assert classifiers.num_classes == classes
    assert len(classifiers.heads) == 8
    assert [head.in_features for head in classifiers.heads] == [2048]*3+[256]*5
    assert all(head.out_features == classes and head.bias is None for head in classifiers.heads)
    assert len({head.weight.data_ptr() for head in classifiers.heads}) == 8
    features = [torch.randn(2, width) for width in [2048]*3+[256]*5]
    assert [tuple(logit.shape) for logit in classifiers(features)] == [(2, classes)]*8


def test_classifier_initializer_and_controlled_routing_gradients(monkeypatch):
    calls = []
    original = nn.init.kaiming_normal_
    def kaiming(weight, *args, **kwargs):
        calls.append((weight.shape, kwargs))
        return original(weight, *args, **kwargs)
    monkeypatch.setattr(nn.init, 'kaiming_normal_', kaiming)
    classifiers = mgn.MGNClassifiers(3)
    assert calls == [(torch.Size([3, width]), {'mode': 'fan_out'}) for width in [2048]*3+[256]*5]
    features = [torch.full((2, width), float(i+2), requires_grad=True) for i, width in enumerate([2048]*3+[256]*5)]
    with torch.no_grad():
        for i, head in enumerate(classifiers.heads):
            head.weight.zero_()
            head.weight[:, 0] = torch.arange(1, 4)*(i+1)
    logits = classifiers(features)
    for i, output in enumerate(logits):
        expected = (torch.arange(1, 4)*(i+1)*(i+2)).float().expand(2, 3)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    sum(logit.sum() for logit in logits).backward()
    for feature, head in zip(features, classifiers.heads):
        assert feature.grad is not None and feature.grad[:, 0].count_nonzero() == 2
        assert head.weight.grad is not None and head.weight.grad.count_nonzero() == head.weight.numel()


@pytest.mark.parametrize('classes', [None, 0, -1, True, 2.5])
def test_classifiers_reject_invalid_source_class_count(classes):
    with pytest.raises(ValueError, match='num_classes'):
        mgn.MGNClassifiers(classes)


@pytest.mark.parametrize('damage', ['reduced_global', 'unreduced_local', 'count', 'batch', 'dtype'])
def test_classifiers_reject_invalid_features(damage):
    classifiers = mgn.MGNClassifiers(3)
    features = [torch.ones(2, width) for width in [2048]*3+[256]*5]
    if damage == 'reduced_global': features[0] = torch.ones(2, 256)
    elif damage == 'unreduced_local': features[-1] = torch.ones(2, 2048)
    elif damage == 'count': features.pop()
    elif damage == 'batch': features[-1] = torch.ones(3, 256)
    else: features[-1] = features[-1].double()
    with pytest.raises(ValueError, match='classifiers'):
        classifiers(features)


@pytest.mark.parametrize('training,batch', [(False, 1), (True, 2)])
def test_public_model_output_order_and_supervision_locations(training, batch):
    from reid.models.outputs import ensure_output_dict
    model = mgn.MGN(5).train(training)
    captured = {}
    handles = [model.pool.register_forward_hook(lambda mod, args, out: captured.update(pooled=out)),
               model.reductions.register_forward_hook(lambda mod, args, out: captured.update(reduced=out)),
               model.classifiers.register_forward_pre_hook(lambda mod, args: captured.update(inputs=args[0]))]
    images = torch.randn(batch, 3, 384, 128)
    with torch.no_grad():
        output = model(images)
    for handle in handles: handle.remove()
    assert ensure_output_dict(output) is output
    assert model.variant == mgn.VARIANT and model.embedding_dim == 2048 and model.feat_dim is None
    assert output['feat_raw'] is output['feat_bn'] is None
    assert output['emb'].shape == (batch, 2048)
    assert [f.shape for f in output['metric_features']] == [torch.Size([batch, 256])]*3
    assert [f.shape for f in output['logits']] == [torch.Size([batch, 5])]*8
    torch.testing.assert_close(output['emb'], torch.cat(captured['reduced'], dim=1), rtol=0, atol=0)
    assert all(a is b for a, b in zip(output['metric_features'], captured['reduced'][:3]))
    for i in range(3):
        torch.testing.assert_close(captured['inputs'][i], captured['pooled'][i].flatten(1), rtol=0, atol=0)
    assert all(a is b for a, b in zip(captured['inputs'][3:], captured['reduced'][3:]))
    assert not torch.allclose(output['emb'].norm(dim=1), torch.ones(batch))
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    if not training:
        with torch.no_grad():
            for head in model.classifiers.heads: head.weight.fill_(10)
            after = model(images)['emb']
        torch.testing.assert_close(output['emb'], after, rtol=0, atol=0)


def test_public_model_real_backward_and_independent_loss_routing():
    from reid.losses.build import build_criterion
    import torch.nn.functional as F
    torch.manual_seed(42)
    model = mgn.MGN(3).train()
    labels = torch.tensor([0, 0, 1, 1])
    output = model(torch.randn(4, 3, 384, 128))
    assert output['emb'].requires_grad
    assert all(feature.requires_grad for feature in output['metric_features'])
    global_ce = sum(F.cross_entropy(head, labels) for head in output['logits'][:3])
    assert all(gradient is None for gradient in torch.autograd.grad(
        global_ce, tuple(model.reductions.parameters()), allow_unused=True, retain_graph=True))
    cfg = {'model': {'head': {}}, 'loss': {
        'id': dict(enabled=True, weight=1., head_aggregation='mean', label_smoothing=0.),
        'triplet': dict(enabled=True, weight=1., margin=1.2, feature_source='model', feature_aggregation='mean'),
        'center': dict(enabled=False)}}
    criterion = build_criterion(cfg, 3, None)
    metric_loss = sum(criterion.triplet(feature, labels) for feature in output['metric_features'])/3
    metric_grads = torch.autograd.grad(metric_loss, tuple(model.reductions.parameters()), allow_unused=True, retain_graph=True)
    for name, gradient in zip(dict(model.reductions.named_parameters()), metric_grads):
        if int(name.split('.')[1]) < 3:
            assert gradient is not None and torch.isfinite(gradient).all()
        else:
            assert gradient is None
    loss, logs = criterion(output, labels)
    manual_ce = sum(F.cross_entropy(head, labels) for head in output['logits'])/8
    torch.testing.assert_close(loss, manual_ce+metric_loss, rtol=0, atol=0)
    assert logs['loss/id'] == pytest.approx(manual_ce.item())
    assert logs['loss/triplet'] == pytest.approx(metric_loss.item())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.backbone.shared.conv1.weight.grad.count_nonzero() > 0


def test_public_model_invalid_geometry_or_single_train_fails_before_bn_mutation():
    model = mgn.MGN(3).train()
    before = {key: value.clone() for key, value in model.named_buffers()}
    for images in [torch.randn(2, 3, 256, 128), torch.randn(1, 3, 384, 128)]:
        with pytest.raises(ValueError): model(images)
        assert all(torch.equal(value, before[key]) for key, value in model.named_buffers())
