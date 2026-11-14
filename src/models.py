"""CIFAR-style ResNet-18 with input normalization, dual BatchNorm and CAM access (CLAUDE.md §4.1).

Architecture follows the DAT official code (models/resnet.py): 3x3 stem conv, stride 1, no max-pool,
BasicBlock x [2,2,2,2], global average pool, linear head. Dual BN = every BN layer (including the
shortcut BN) has two parameter sets, selected per forward call: 'A' (clean / main branch) or 'B'
(frequency-perturbed branch). Test-time inference always uses 'A'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    """Per-channel (x - mean) / std as the model's first layer, so attacks act on [0,1] images (§4.1)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class DualBN2d(nn.Module):
    """BatchNorm with two parameter/statistic sets 'A' and 'B'. With dual=False it is a plain BN (A only)."""

    def __init__(self, num_features, dual=True):
        super().__init__()
        self.bn_a = nn.BatchNorm2d(num_features)
        self.bn_b = nn.BatchNorm2d(num_features) if dual else None
        self.route = "A"

    def forward(self, x):
        if self.route == "B" and self.bn_b is not None:
            return self.bn_b(x)
        return self.bn_a(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = DualBN2d(planes, dual)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = DualBN2d(planes, dual)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)
            self.shortcut_bn = DualBN2d(planes, dual)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = x if self.shortcut is None else self.shortcut_bn(self.shortcut(x))
        return F.relu(out + sc)


def _init_weights(m):
    """Same initialisation as the DAT official code (kaiming conv, xavier linear, BN 1/0); shared by all methods."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ResNet18(nn.Module):
    def __init__(self, num_classes=43, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), dual_bn=True):
        super().__init__()
        self.dual_bn = dual_bn
        self.normalize = Normalize(mean, std)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = DualBN2d(64, dual_bn)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)
        self.apply(_init_weights)
        self._bns = [m for m in self.modules() if isinstance(m, DualBN2d)]

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s, self.dual_bn))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def set_bn(self, route: str):
        assert route in ("A", "B")
        for m in self._bns:
            m.route = route

    def features(self, x, bn="A"):
        """Final feature map F(v): 512 x 4 x 4 for a 32x32 input (§5.6 L_att)."""
        self.set_bn(bn)
        out = F.relu(self.bn1(self.conv1(self.normalize(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return out

    def forward(self, x, bn="A", return_feat=False):
        feat = self.features(x, bn)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        logits = self.linear(pooled)
        self.set_bn("A")  # leave the model in test-time configuration
        if return_feat:
            return logits, feat
        return logits

    def cam(self, feat, y):
        """CAM(v) = W_y^T F(v) [adaptation of FreqDebias Eq. 4-6]: for GAP + linear head the class
        activation map is the feature map weighted by the head row of class y. Returns B x 16 (flattened 4x4)."""
        w = self.linear.weight[y]                      # B x 512
        cam = torch.einsum("bchw,bc->bhw", feat.float(), w.float())
        return cam.flatten(1)


def build_model(cfg: dict, mean, std, num_classes=43):
    return ResNet18(num_classes=num_classes, mean=mean, std=std, dual_bn=bool(cfg.get("dual_bn", False)))
