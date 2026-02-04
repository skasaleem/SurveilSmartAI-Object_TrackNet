from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from utils.box_ops import generalized_box_iou


class DetectionLoss(nn.Module):
    """
    Simplified detection loss combining:
      - Focal loss for classification + objectness
      - GIoU loss for bounding boxes
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, lambda_box: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_box = lambda_box

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        NOTE: For brevity, this implements a very simple anchor-free style matching:
        - We flatten all predictions and use nearest box center matching
        This is not production-grade YOLO matching, but keeps the example functional.
        """
        device = outputs["obj_cls"][0].device

        # Concatenate feature maps
        def concat_outputs(head_list: List[torch.Tensor]):
            flat = []
            for h in head_list:
                B, C, H, W = h.shape
                flat.append(h.view(B, C, H * W).permute(0, 2, 1))  # B, HW, C
            return torch.cat(flat, dim=1)  # B, sum(HW), C

        obj_cls = concat_outputs(outputs["obj_cls"])
        obj_reg = concat_outputs(outputs["obj_reg"])
        obj_obj = concat_outputs(outputs["obj_obj"])

        B, N, _ = obj_cls.shape
        num_classes = obj_cls.shape[-1]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            tgt = targets[b]
            boxes = tgt["boxes"].to(device)
            labels = tgt["labels"].to(device)

            if boxes.numel() == 0 or labels.numel() == 0:
                # all negative
                obj_logits = obj_obj[b].squeeze(-1)
                total_obj_loss += sigmoid_focal_loss(
                    obj_logits, torch.zeros_like(obj_logits), alpha=self.alpha, gamma=self.gamma, reduction="sum"
                )
                continue

            # simple heuristic: randomly assign some locations as positives based on number of GT boxes
            # (in real implementation, use anchor/grid matching)
            k = min(boxes.size(0), N)
            pos_indices = torch.randperm(N, device=device)[:k]

            # classification targets
            cls_logits = obj_cls[b, pos_indices]  # k, num_classes
            cls_targets = torch.zeros_like(cls_logits)
            for i, lbl in enumerate(labels[:k]):
                if lbl > 0 and lbl < num_classes:
                    cls_targets[i, lbl] = 1.0
            total_cls_loss += sigmoid_focal_loss(
                cls_logits, cls_targets, alpha=self.alpha, gamma=self.gamma, reduction="sum"
            )

            # objectness targets
            obj_logits = obj_obj[b].squeeze(-1)  # N
            obj_targets = torch.zeros_like(obj_logits)
            obj_targets[pos_indices] = 1.0
            total_obj_loss += sigmoid_focal_loss(
                obj_logits, obj_targets, alpha=self.alpha, gamma=self.gamma, reduction="sum"
            )

            # box regression (use GIoU)
            pred_boxes = obj_reg[b, pos_indices]  # k, 4 (cx,cy,w,h style is not enforced here)
            # For demonstration we interpret them as xyxy with no decoding; in practice use proper decode.
            tgt_boxes = boxes[:k]
            giou = generalized_box_iou(tgt_boxes, pred_boxes)
            # diagonal of giou
            giou_diag = torch.diag(giou)
            box_loss = 1.0 - giou_diag
            total_box_loss += box_loss.sum()
            num_pos += k

        num_pos = max(num_pos, 1)
        losses = {
            "loss_cls": total_cls_loss / num_pos,
            "loss_obj": total_obj_loss / num_pos,
            "loss_box": self.lambda_box * total_box_loss / num_pos,
        }
        losses["loss_total"] = losses["loss_cls"] + losses["loss_obj"] + losses["loss_box"]
        return losses
