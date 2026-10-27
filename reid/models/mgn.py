"""MGN v3 topology; reference details pinned by the project variant.

Wang et al. arXiv:1804.01438v3; seathiefwang/MGN-pytorch at
384e82002e40b71157506e23d71e6709fa830757. Architecture-local components
never download weights and never infer class counts from evaluation datasets.
"""
import copy
import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

VARIANT = 'paper_v3_global2048_independent_v1'
ARTIFACT_ID = 'resnet50-19c8e357'
IMAGENET_SHA256 = '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097'
HEAD_INIT_ID = 'reference_a_384e820_bias_free_v1'


def _read_historical_weights(weights_path=None):
    path = (Path(weights_path) if weights_path is not None else
            Path(torch.hub.get_dir()) / 'checkpoints' / f'{ARTIFACT_ID}.pth')
    # Verify and deserialize the same stream. This pinned official artifact uses
    # legacy tar serialization; no unverified bytes enter weights_only=False.
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != IMAGENET_SHA256:
            raise ValueError(f'MGN ImageNet artifact SHA256 mismatch: {path}')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=False)


def _load_imagenet_state(base, state):
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError('ImageNet state must be a string-keyed tensor mapping.')
    expected = base.state_dict()
    counters = {k for k in expected if k.endswith('.num_batches_tracked')}
    missing = set(expected) - counters - set(state)
    unexpected = set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f'ImageNet state keys incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key, tensor in state.items():
        if (not torch.is_tensor(tensor) or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype):
            raise ValueError(f'ImageNet tensor incompatible: {key}')
    # Validate fc too, then discard it when constructing the feature backbone.
    base.load_state_dict({k: state[k] if k in state else torch.zeros_like(v)
                          for k, v in expected.items()}, strict=True)


class MGNBackbone(nn.Module):
    """Shared res_conv4_1, then three independent residual branches."""
    def __init__(self, *, pretrained=False, weights_path=None):
        super().__init__()
        if weights_path is not None and not pretrained:
            raise ValueError('weights_path requires pretrained=True')
        base = resnet50(weights=None)
        if pretrained:
            _load_imagenet_state(base, _read_historical_weights(weights_path))
        self.shared = nn.Sequential(OrderedDict([
            ('conv1', base.conv1), ('bn1', base.bn1), ('relu', base.relu),
            ('maxpool', base.maxpool), ('layer1', base.layer1), ('layer2', base.layer2),
            ('layer3_0', base.layer3[0]),
        ]))
        tail = nn.Sequential(*list(base.layer3.children())[1:])
        self.branches = nn.ModuleList([
            nn.Sequential(OrderedDict([('layer3', copy.deepcopy(tail)),
                                       ('layer4', copy.deepcopy(base.layer4))]))
            for _ in range(3)
        ])
        for branch in self.branches[1:]:
            first = branch.layer4[0]
            first.conv2.stride = first.downsample[0].stride = (1, 1)
            first.stride = 1

    @staticmethod
    def validate_input(images):
        if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] <= 0
                or tuple(images.shape[1:]) != (3, 384, 128) or not images.is_floating_point()):
            raise ValueError('MGN requires floating images [B,3,384,128] with B > 0.')

    def forward(self, images):
        self.validate_input(images)
        shared = self.shared(images)
        return tuple(branch(shared) for branch in self.branches)


def _validate_features(features, shapes, name):
    if not isinstance(features, (list, tuple)) or len(features) != len(shapes):
        raise ValueError(f'{name} requires exactly {len(shapes)} feature tensors.')
    first = None
    for feature, shape in zip(features, shapes):
        if (not torch.is_tensor(feature) or not feature.is_floating_point()
                or feature.ndim != len(shape)+1 or tuple(feature.shape[1:]) != tuple(shape)
                or feature.shape[0] <= 0):
            raise ValueError(f'{name} feature must have shape [B,{",".join(map(str, shape))}] with B > 0.')
        if first is not None and (feature.shape[0] != first.shape[0]
                or feature.dtype != first.dtype or feature.device != first.device):
            raise ValueError(f'{name} requires matching batch size, dtype and device.')
        first = feature


class MGNMaxPool(nn.Module):
    """Three whole-map maxima, then P2 top/bottom and P3 top/middle/bottom."""
    def forward(self, maps):
        _validate_features(maps, [(2048, 12, 4), (2048, 24, 8), (2048, 24, 8)], 'MGN pooling')
        parts = list(maps)
        parts.extend(maps[1].chunk(2, dim=2))
        parts.extend(maps[2].chunk(3, dim=2))
        return tuple(F.max_pool2d(part, kernel_size=part.shape[-2:]) for part in parts)


class MGNReductions(nn.Module):
    """Eight identical-in-value, independent Conv/BN/ReLU reduction copies."""
    def __init__(self):
        super().__init__()
        template = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                 nn.BatchNorm2d(256, eps=1e-5, momentum=.1),
                                 nn.ReLU(inplace=False))
        nn.init.kaiming_normal_(template[0].weight, mode='fan_in')
        nn.init.normal_(template[1].weight, 1., .02)
        nn.init.zeros_(template[1].bias)
        nn.init.zeros_(template[1].running_mean)
        nn.init.ones_(template[1].running_var)
        nn.init.zeros_(template[1].num_batches_tracked)
        self.parts = nn.ModuleList([copy.deepcopy(template) for _ in range(8)])

    def forward(self, pooled):
        _validate_features(pooled, [(2048, 1, 1)]*8, 'MGN reductions')
        if self.training and pooled[0].shape[0] == 1:
            raise ValueError('MGN reduction training BatchNorm requires B > 1.')
        return tuple(reduction(feature).flatten(1) for reduction, feature in zip(self.parts, pooled))
