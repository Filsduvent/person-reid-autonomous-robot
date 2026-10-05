"""Backbone, stripe pooling and independent PCB reductions; no identity heads.

Reference: huanghoujing/beyond-part-models @
1686e889eb01c28a54b633051418012e15d9c9f3, bpm/model/resnet.py
(Bottleneck:56–92, ResNet:95–147, resnet50:182–190) and
bpm/model/PCBModel.py:20–23. Only stride=1, dilation=1 is implemented.
"""

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


HISTORICAL_IMAGENET_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
HISTORICAL_IMAGENET_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"


def _read_historical_weights(weights_path=None):
    """Read only the checksum-pinned reference artifact, including cached files.

    This exact official artifact uses PyTorch's legacy tar format, which cannot
    be read with weights_only=True. Never deserialize a different file through
    the legacy loader: the full SHA256 check must precede torch.load.
    """
    if weights_path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-19c8e357.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                HISTORICAL_IMAGENET_URL, str(path),
                hash_prefix=HISTORICAL_IMAGENET_SHA256, progress=False,
            )
    else:
        path = Path(weights_path)
    # Hash and deserialize the same open file, also validating pre-existing cache.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != HISTORICAL_IMAGENET_SHA256:
            raise ValueError(f"Historical PCB ImageNet weight SHA256 mismatch: {path}")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=False)


class PCBBackbone(nn.Module):
    """ResNet50 through layer4: [B,3,384,128] -> [B,2048,24,8].

    This is an internal feature component, not a registered ReID model. Use
    pretrained=False for offline unit tests and future trained-state loading;
    pretrained=True opts into the pinned historical ImageNet initialization.
    weights_path optionally supplies that exact artifact locally.
    """

    def __init__(self, *, pretrained=False, weights_path=None):
        super().__init__()
        if weights_path is not None and not pretrained:
            raise ValueError("weights_path requires pretrained=True")
        # torchvision matches Huang's [3,4,6,3] bottleneck topology, conv2 stride,
        # fan-out normal convolution initialization, and unit/zero backbone BN.
        # Never use a torchvision pretrained enum (the baseline uses V2).
        base = resnet50(weights=None)
        first = base.layer4[0]
        first.conv2.stride = (1, 1)
        first.downsample[0].stride = (1, 1)
        first.stride = 1
        # Dilation remains 1; do not replace stride with dilation.
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            self.add_module(name, getattr(base, name))
        # avgpool/fc are not registered or executed by this component.
        if pretrained:
            self.load_imagenet_state_dict(_read_historical_weights(weights_path))

    def load_imagenet_state_dict(self, state):
        """Validate ImageNet tensors before loading; only fc is discarded.

        Old PyTorch has no BN num_batches_tracked buffers. Insert zero counters
        explicitly for that known compatibility difference. All other missing,
        unexpected, shape- or dtype-incompatible tensors are errors.
        """
        if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
            raise ValueError("ImageNet state must be a string-keyed tensor mapping")
        expected = self.state_dict()
        fc_shapes = {"fc.weight": (1000, 2048), "fc.bias": (1000,)}
        counters = {key for key in expected if key.endswith(".num_batches_tracked")}
        required = (set(expected) - counters) | set(fc_shapes)
        missing = required - set(state)
        unexpected = set(state) - set(expected) - set(fc_shapes)
        if missing or unexpected:
            raise ValueError(f"ImageNet state keys incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for key, value in state.items():
            shape = fc_shapes[key] if key in fc_shapes else tuple(expected[key].shape)
            dtype = torch.float32 if key in fc_shapes else expected[key].dtype
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"ImageNet tensor incompatible: {key}; expected shape={shape}, dtype={dtype}")
        mapped = {key: state[key] if key in state else torch.zeros_like(value)
                  for key, value in expected.items()}
        self.load_state_dict(mapped, strict=True)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class PCBStripePool(nn.Module):
    """Six equal horizontal stripes, pooled top-to-bottom without flattening.

    Consumes [B,C,H,W] with positive H divisible by six and positive W.
    The standard backbone map [B,2048,24,8] yields six [B,2048,1,1]
    tensors. Matches pinned PCBModel.py:50–58; the stripe count is fixed
    by the selected architecture, not exposed as an experiment setting.
    """

    @staticmethod
    def partition(feature_map):
        """Return six full-width views in spatial order, before pooling."""
        if not torch.is_tensor(feature_map) or feature_map.ndim != 4:
            raise ValueError("PCB stripe pooling requires a 4D [B,C,H,W] tensor")
        height, width = feature_map.shape[-2:]
        if height <= 0 or height % 6 != 0:
            raise ValueError("PCB feature-map height must be positive and divisible by 6")
        if width <= 0:
            raise ValueError("PCB feature-map width must be positive")
        stripe_height = height // 6
        return tuple(feature_map[:, :, i * stripe_height:(i + 1) * stripe_height, :]
                     for i in range(6))

    def forward(self, feature_map):
        return tuple(F.avg_pool2d(stripe, kernel_size=stripe.shape[-2:])
                     for stripe in self.partition(feature_map))


class PCBPartReductions(nn.Module):
    """Six independent Conv/BN/ReLU reductions, returning ordered [B,256] parts.

    Matches pinned PCBModel.py:26–32,60–63. Initialization explicitly follows
    PyTorch v0.3.0 modules/conv.py:_ConvNd.reset_parameters and
    modules/batchnorm.py:_BatchNorm.reset_parameters, not modern BN defaults.
    This component owns only reductions; it never initializes a backbone.
    """

    def __init__(self):
        super().__init__()
        self.local_conv_list = nn.ModuleList()
        for _ in range(6):
            conv = nn.Conv2d(2048, 256, kernel_size=1, bias=True)
            bn = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1,
                                affine=True, track_running_stats=True)
            # Override constructor defaults only on these newly created layers.
            bound = 1 / math.sqrt(2048)
            nn.init.uniform_(conv.weight, -bound, bound)
            nn.init.uniform_(conv.bias, -bound, bound)
            nn.init.uniform_(bn.weight, 0, 1)
            nn.init.zeros_(bn.bias)
            nn.init.zeros_(bn.running_mean)
            nn.init.ones_(bn.running_var)
            nn.init.zeros_(bn.num_batches_tracked)
            self.local_conv_list.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))

    def forward(self, pooled_parts):
        if not isinstance(pooled_parts, (tuple, list)) or len(pooled_parts) != 6:
            raise ValueError("PCB reductions require exactly six pooled stripe tensors")
        # Validate the complete input before any training-mode BN state changes.
        batch_size = None
        for part in pooled_parts:
            if (not torch.is_tensor(part) or part.ndim != 4
                    or tuple(part.shape[1:]) != (2048, 1, 1) or part.shape[0] <= 0):
                raise ValueError("Each PCB pooled stripe must have shape [B,2048,1,1] with B > 0")
            if batch_size is not None and part.shape[0] != batch_size:
                raise ValueError("PCB pooled stripes must have equal batch sizes")
            batch_size = part.shape[0]
        return tuple(reduction(part).flatten(1)
                     for reduction, part in zip(self.local_conv_list, pooled_parts))
