import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.objtracknet import ObjTrackNet
from utils.transforms import ToTensor, Resize


def decode_predictions(
    obj_cls_list,
    obj_reg_list,
    obj_obj_list,
    conf_threshold: float = 0.3,
    obj_threshold: float = 0.3,
    img_size: int = 640,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very simple decoding:
      - Concatenate all feature maps
      - Apply sigmoid to class & objectness
      - Treat reg outputs as (x1,y1,x2,y2) in image coordinates
    """
    device = obj_cls_list[0].device

    def concat_outputs(head_list):
        flat = []
        for h in head_list:
            B, C, H, W = h.shape
            flat.append(h.view(B, C, H * W).permute(0, 2, 1))  # B, HW, C
        return torch.cat(flat, dim=1)  # B, N, C

    obj_cls = concat_outputs(obj_cls_list)
    obj_reg = concat_outputs(obj_reg_list)
    obj_obj = concat_outputs(obj_obj_list)

    # single image only
    obj_cls = obj_cls[0]
    obj_reg = obj_reg[0]
    obj_obj = obj_obj[0]

    obj_scores = torch.sigmoid(obj_obj).squeeze(-1)  # N
    cls_scores = torch.sigmoid(obj_cls)              # N, num_classes

    # filter by objectness
    keep = obj_scores > obj_threshold
    if keep.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    obj_scores = obj_scores[keep]
    cls_scores = cls_scores[keep]
    boxes = obj_reg[keep]  # N, 4

    # choose best class per prediction
    cls_max_scores, cls_ids = cls_scores.max(dim=-1)
    scores = obj_scores * cls_max_scores

    keep2 = scores > conf_threshold
    boxes = boxes[keep2]
    scores = scores[keep2]

    # clamp boxes to image size
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_size - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_size - 1)

    return boxes, scores


def run_on_image(image_path: str, ckpt_path: str, img_size: int = 640, device: str = "cuda"):
    model = ObjTrackNet(num_classes=10, num_human_classes=1, base_channels=32)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    orig = image_bgr.copy()
    transform = Resize((img_size, img_size))
    to_tensor = ToTensor()
    img_tensor, _ = to_tensor(image_bgr, {})
    img_tensor, _ = transform(img_tensor, {})
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        boxes, scores = decode_predictions(outputs["obj_cls"], outputs["obj_reg"], outputs["obj_obj"])

    # draw boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = os.path.splitext(image_path)[0] + "_det.jpg"
    cv2.imwrite(out_path, orig)
    print(f"Saved result to {out_path}")


def run_on_video(video_path: str, ckpt_path: str, img_size: int = 640, device: str = "cuda"):
    model = ObjTrackNet(num_classes=10, num_human_classes=1, base_channels=32)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(video_path)[0] + "_det.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    to_tensor = ToTensor()
    resize = Resize((img_size, img_size))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, _ = to_tensor(frame, {})
        img_tensor, _ = resize(img_tensor, {})
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            boxes, scores = decode_predictions(outputs["obj_cls"], outputs["obj_reg"], outputs["obj_obj"])

        vis = frame.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Saved detection video to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference demo for ObjTrackNet")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image, args.ckpt, args.img_size, args.device)
    elif args.video:
        run_on_video(args.video, args.ckpt, args.img_size, args.device)
    else:
        print("Please provide either --image or --video")


if __name__ == "__main__":
    main()
