from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Lightweight Transformer encoder layer applied on flattened spatial features.
    """

    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, H, W -> B, HW, C
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # B, HW, C

        attn_output, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = x_flat + self.dropout1(attn_output)
        x_flat = self.norm1(x_flat)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x_flat))))
        x_flat = x_flat + self.dropout2(ff)
        x_flat = self.norm2(x_flat)

        x_out = x_flat.transpose(1, 2).view(B, C, H, W)
        return x_out


class HybridBackbone(nn.Module):
    """
    Simple hybrid Conv + Transformer backbone that outputs three feature levels
    similar to P3, P4, P5 with strides 8, 16, 32.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Stem
        self.stem = nn.Sequential(
            ConvBlock(in_channels, c, 3, 2, 1),  # 1/2
            ConvBlock(c, c, 3, 1, 1),
            ConvBlock(c, c * 2, 3, 2, 1),       # 1/4
        )

        # Stage 2 (P3, stride 8)
        self.stage2 = nn.Sequential(
            ConvBlock(c * 2, c * 2, 3, 1, 1),
            ConvBlock(c * 2, c * 4, 3, 2, 1),   # 1/8
        )
        self.trans2 = TransformerEncoderLayer(d_model=c * 4, nhead=4, dim_feedforward=c * 8)

        # Stage 3 (P4, stride 16)
        self.stage3 = nn.Sequential(
            ConvBlock(c * 4, c * 4, 3, 1, 1),
            ConvBlock(c * 4, c * 8, 3, 2, 1),   # 1/16
        )
        self.trans3 = TransformerEncoderLayer(d_model=c * 8, nhead=4, dim_feedforward=c * 16)

        # Stage 4 (P5, stride 32)
        self.stage4 = nn.Sequential(
            ConvBlock(c * 8, c * 8, 3, 1, 1),
            ConvBlock(c * 8, c * 16, 3, 2, 1),  # 1/32
        )
        self.trans4 = TransformerEncoderLayer(d_model=c * 16, nhead=4, dim_feedforward=c * 32)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage2(x)
        p3 = self.trans2(p3)

        p4 = self.stage3(p3)
        p4 = self.trans3(p4)

        p5 = self.stage4(p4)
        p5 = self.trans4(p5)

        return {"P3": p3, "P4": p4, "P5": p5}
