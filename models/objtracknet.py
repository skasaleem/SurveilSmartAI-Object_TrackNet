from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .backbone_hybrid import HybridBackbone
from .bifpn import BiFPN


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 1, 1, 0)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors * 1, 1, 1, 0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []
        for f in feats:
            cls = self.cls_conv(f)
            reg = self.reg_conv(f)
            cls_outputs.append(self.cls_pred(cls))
            reg_outputs.append(self.reg_pred(reg))
            obj_outputs.append(self.obj_pred(reg))
        return cls_outputs, reg_outputs, obj_outputs


class ObjTrackNet(nn.Module):
    """
    Simplified ObjTrackNet with dual detection heads (object + human).
    """

    def __init__(self, num_classes: int, num_human_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.backbone = HybridBackbone(in_channels=3, base_channels=base_channels)
        self.neck = BiFPN(
            in_channels={"P3": base_channels * 4, "P4": base_channels * 8, "P5": base_channels * 16},
            out_channels=128,
            num_layers=2,
        )

        self.obj_head = DetectionHead(128, num_classes)
        self.human_head = DetectionHead(128, num_human_classes)

    def forward(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        images: B x 3 x H x W
        Returns dict with:
          - obj_cls, obj_reg, obj_obj
          - human_cls, human_reg, human_obj
        """
        feats = self.backbone(images)  # dict P3,P4,P5
        fused = self.neck(feats)
        feat_list = [fused["P3"], fused["P4"], fused["P5"]]

        obj_cls, obj_reg, obj_obj = self.obj_head(feat_list)
        hum_cls, hum_reg, hum_obj = self.human_head(feat_list)

        return {
            "obj_cls": obj_cls,
            "obj_reg": obj_reg,
            "obj_obj": obj_obj,
            "hum_cls": hum_cls,
            "hum_reg": hum_reg,
            "hum_obj": hum_obj,
        }


if __name__ == "__main__":
    # quick sanity check
    model = ObjTrackNet(num_classes=10, num_human_classes=1)
    x = torch.randn(2, 3, 640, 640)
    out = model(x)
    for k, v in out.items():
        print(k, [t.shape for t in v])
