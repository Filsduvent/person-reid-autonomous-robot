import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ReidBaseline(nn.Module):
    def __init__(self, pretrained: bool, last_conv_stride: int, embedding_dim: int, bnneck: bool, normalize: bool):
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

        self.bnneck = nn.BatchNorm1d(embedding_dim) if bnneck else None
        if self.bnneck is not None:
            nn.init.constant_(self.bnneck.weight, 1.0)
            nn.init.constant_(self.bnneck.bias, 0.0)

        self.normalize = bool(normalize)

    def forward(self, x):
        feat_map = self.backbone(x)
        feat = self.gap(feat_map).flatten(1)            # [N, 2048]
        emb = self.embedding(feat)                      # [N, D]
        if self.bnneck is not None:
            emb = self.bnneck(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb
