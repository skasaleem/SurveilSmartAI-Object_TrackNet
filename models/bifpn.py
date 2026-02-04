from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FastBiFPNLayer(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.eps = 1e-4
        self.num_channels = num_channels

        # top-down weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # bottom-up weights
        self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w4 = nn.Parameter(torch.ones(3, dtype=torch.float32))

        self.conv6_up = SeparableConvBlock(num_channels, num_channels)
        self.conv5_up = SeparableConvBlock(num_channels, num_channels)
        self.conv4_up = SeparableConvBlock(num_channels, num_channels)

        self.conv4_down = SeparableConvBlock(num_channels, num_channels)
        self.conv5_down = SeparableConvBlock(num_channels, num_channels)
        self.conv6_down = SeparableConvBlock(num_channels, num_channels)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        P3, P4, P5 = feats["P3"], feats["P4"], feats["P5"]

        # ensure same channels
        # assume input already has num_channels

        # Top-down pathway
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.eps)
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.eps)

        P5_td = P5
        P4_td = self.conv5_up(w1[0] * P4 + w1[1] * F.interpolate(P5_td, size=P4.shape[-2:], mode="nearest"))
        P3_td = self.conv4_up(w2[0] * P3 + w2[1] * F.interpolate(P4_td, size=P3.shape[-2:], mode="nearest"))

        # Bottom-up pathway
        w3 = F.relu(self.w3)
        w3 = w3 / (torch.sum(w3) + self.eps)
        w4 = F.relu(self.w4)
        w4 = w4 / (torch.sum(w4) + self.eps)

        P3_out = P3_td
        P4_out = self.conv4_down(
            w3[0] * P4 + w3[1] * P4_td + w3[2] * F.max_pool2d(P3_out, kernel_size=2)
        )
        P5_out = self.conv5_down(
            w4[0] * P5 + w4[1] * P5_td + w4[2] * F.max_pool2d(P4_out, kernel_size=2)
        )

        return {"P3": P3_out, "P4": P4_out, "P5": P5_out}


class BiFPN(nn.Module):
    def __init__(self, in_channels: Dict[str, int], out_channels: int = 128, num_layers: int = 2):
        super().__init__()
        self.out_channels = out_channels

        # lateral convs to unify channels
        self.lateral_convs = nn.ModuleDict()
        for level, c in in_channels.items():
            self.lateral_convs[level] = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)

        self.bifpn_layers = nn.ModuleList([FastBiFPNLayer(out_channels) for _ in range(num_layers)])

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = {k: self.lateral_convs[k](v) for k, v in feats.items()}
        for layer in self.bifpn_layers:
            feats = layer(feats)
        return feats
