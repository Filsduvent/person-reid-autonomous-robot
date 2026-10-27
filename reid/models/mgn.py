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


def _validate_num_classes(num_classes):
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError('MGN num_classes must be a positive source identity count.')


class MGNClassifiers(nn.Module):
    """Three unreduced global 2048-D and five reduced local 256-D CE heads."""
    def __init__(self, num_classes):
        super().__init__()
        _validate_num_classes(num_classes)
        self.num_classes = num_classes
        self.heads = nn.ModuleList([nn.Linear(width, num_classes, bias=False)
                                    for width in [2048]*3+[256]*5])
        for head in self.heads:
            nn.init.kaiming_normal_(head.weight, mode='fan_out')

    def forward(self, features):
        _validate_features(features, [(2048,)]*3+[(256,)]*5, 'MGN classifiers')
        return tuple(head(feature) for head, feature in zip(self.heads, features))


class MGN(nn.Module):
    """Frozen paper-v3 MGN, exposing raw retrieval and explicit metric features."""
    variant = VARIANT
    embedding_dim = 2048
    feat_dim = None

    def __init__(self, num_classes, *, pretrained=False, weights_path=None):
        super().__init__()
        _validate_num_classes(num_classes)
        self.num_classes = num_classes
        self.backbone = MGNBackbone(pretrained=pretrained, weights_path=weights_path)
        self.pool = MGNMaxPool()
        self.reductions = MGNReductions()
        self.classifiers = MGNClassifiers(num_classes)

    def forward(self, images):
        self.backbone.validate_input(images)
        if self.training and images.shape[0] == 1:
            raise ValueError('MGN training reduction BatchNorm requires B > 1.')
        pooled = self.pool(self.backbone(images))
        reduced = self.reductions(pooled)
        classifier_features = tuple(part.flatten(1) for part in pooled[:3]) + reduced[3:]
        return dict(emb=torch.cat(reduced, dim=1), feat_raw=None, feat_bn=None,
                    metric_features=reduced[:3], logits=self.classifiers(classifier_features))


def validate_mgn_model_config(model_cfg):
    """Canonicalize the sole supported variant/provenance without artifact I/O."""
    allowed = {'name', 'variant', 'pretrained', 'weights_path', 'initialization', 'head'}
    if not isinstance(model_cfg, dict) or model_cfg.get('name') != 'mgn':
        raise ValueError('MGN configuration requires model.name=mgn.')
    if set(model_cfg)-allowed:
        raise ValueError(f'Unsupported MGN model fields: {sorted(set(model_cfg)-allowed)}')
    if model_cfg.get('variant', VARIANT) != VARIANT:
        raise ValueError(f'MGN model.variant must be {VARIANT}.')
    pretrained = model_cfg.get('pretrained', True)
    if type(pretrained) is not bool:
        raise ValueError('MGN model.pretrained must be boolean.')
    path = model_cfg.get('weights_path')
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError('MGN weights_path must be a nonempty string or None.')
    if path is not None and not pretrained:
        raise ValueError('MGN weights_path requires pretrained=true.')
    head = model_cfg.get('head', {})
    if not isinstance(head, dict) or set(head)-{'classifier'} or head.get('classifier', True) is not True:
        raise ValueError('MGN head supports only classifier=true; topology is fixed by variant.')
    initialization = model_cfg.get('initialization', {})
    expected = dict(artifact_id=ARTIFACT_ID, sha256=IMAGENET_SHA256, head_init_id=HEAD_INIT_ID)
    if not isinstance(initialization, dict) or set(initialization)-set(expected):
        raise ValueError('Unsupported MGN initialization fields.')
    for key, value in expected.items():
        if initialization.get(key, value) != value:
            raise ValueError(f'MGN initialization.{key} must be {value}.')
    model_cfg.update(variant=VARIANT, pretrained=pretrained,
                     head={'classifier': True}, initialization=expected)
