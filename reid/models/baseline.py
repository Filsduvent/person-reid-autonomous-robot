import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ReidBaseline(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        last_conv_stride: int,
        embedding_dim: int,
        bnneck: bool,
        normalize: bool,
        metric_feat: str = "bn",
        num_classes: int | None = None,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights)

        # Adjust last stage stride if requested (stride in layer4[0].conv2 and downsample)
        if last_conv_stride == 1:
            m.layer4[0].conv2.stride = (1, 1)
            m.layer4[0].downsample[0].stride = (1, 1)

        # Remove classifier head
        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        in_dim = 2048
        self.embedding = nn.Identity() if embedding_dim == in_dim else nn.Linear(in_dim, embedding_dim, bias=False)

        if metric_feat not in {"raw", "bn"}:
            raise ValueError(f"Unsupported metric feature '{metric_feat}'. Use 'raw' or 'bn'.")

        self.bnneck = nn.BatchNorm1d(embedding_dim) if bnneck else None
        if self.bnneck is not None:
            nn.init.constant_(self.bnneck.weight, 1.0)
            nn.init.constant_(self.bnneck.bias, 0.0)

        self.normalize = bool(normalize)
        self.metric_feat = metric_feat
        self.classifier = None
        if num_classes is not None and int(num_classes) > 0:
            self.classifier = nn.Linear(embedding_dim, int(num_classes), bias=False)

    def forward(self, x):
        feat_map = self.backbone(x)
        feat = self.gap(feat_map).flatten(1)            # [N, 2048]
        feat_raw = self.embedding(feat)                 # [N, D] (pre-BN)
        if self.bnneck is not None:
            feat_bn = self.bnneck(feat_raw)
        else:
            feat_bn = feat_raw

        metric_base = feat_bn if self.metric_feat == "bn" else feat_raw
        emb = F.normalize(metric_base, p=2, dim=1) if self.normalize else metric_base
        logits = self.classifier(feat_bn) if self.classifier is not None else None

        return {
            "feat_raw": feat_raw,
            "feat_bn": feat_bn,
            "emb": emb,
            "logits": logits,
        }
