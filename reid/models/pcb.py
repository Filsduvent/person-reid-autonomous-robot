"""Feature backbone for Huang's independent-part PCB; no PCB heads yet.

Reference: huanghoujing/beyond-part-models @
1686e889eb01c28a54b633051418012e15d9c9f3, bpm/model/resnet.py
(Bottleneck:56–92, ResNet:95–147, resnet50:182–190) and
bpm/model/PCBModel.py:20–23. Only stride=1, dilation=1 is implemented.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
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
