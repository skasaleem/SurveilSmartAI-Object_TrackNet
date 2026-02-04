import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.objtracknet import ObjTrackNet
from losses.detection_loss import DetectionLoss
from utils.transforms import ToTensor, Resize


class FrameFolderDataset(Dataset):
    """
    Simple dataset that reads all images from a folder (recursively).
    Used for deployment / latency-aware fine-tuning where we mostly
    want the model to see deployment-like data distribution.

    If no labels are available, we can fine-tune only latency-related
    components or use pseudo-labels. For now, we treat all images as
    background-only (no GT boxes).
    """

    def __init__(self, root: str, img_size: int = 640):
        self.img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.img_paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.img_paths = sorted(self.img_paths)
        if not self.img_paths:
            raise RuntimeError(f"No images found in {root}")
        self.to_tensor = ToTensor()
        self.resize = Resize((img_size, img_size))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        path = self.img_paths[index]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_tensor, _ = self.to_tensor(img, {})
        img_tensor, _ = self.resize(img_tensor, {})
        # dummy target: background only
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return img_tensor, target


def latency_penalty(latency_ms: float, target_ms: float) -> float:
    """
    Simple latency penalty:
      L_lat = max(0, (latency_ms - target_ms) / target_ms)
    """
    return max(0.0, (latency_ms - target_ms) / max(target_ms, 1e-3))


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 latency-aware fine-tuning")
    parser.add_argument("--data-root", type=str, required=True, help="Folder with deployment-like frames")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint from Stage 2 (MOT)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-latency-ms", type=float, default=50.0, help="Target per-batch latency (ms)")
    parser.add_argument("--lambda-lat", type=float, default=0.1, help="Weight for latency loss")
    parser.add_argument("--out-dir", type=str, default="checkpoints_deploy")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = FrameFolderDataset(args.data_root, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ObjTrackNet(num_classes=10, num_human_classes=1, base_channels=32).to(device)

    # Load Stage 2 checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)

    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch} [deploy]", ncols=120)
        for images, targets in pbar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()

            losses: Dict[str, torch.Tensor] = criterion(outputs, targets)
            base_loss = losses["loss_total"]

            batch_latency_ms = (t1 - t0) * 1000.0
            lat_pen = latency_penalty(batch_latency_ms, args.target_latency_ms)
            lat_loss = torch.tensor(lat_pen, dtype=torch.float32, device=device)

            total_batch_loss = base_loss + args.lambda_lat * lat_loss
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            pbar.set_postfix({
                "loss": f"{total_batch_loss.item():.4f}",
                "base": f"{base_loss.item():.4f}",
                "lat": f"{lat_loss.item():.4f}",
                "lat_ms": f"{batch_latency_ms:.1f}",
            })

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        ckpt_path = Path(args.out_dir) / f"objtracknet_deploy_epoch{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "avg_loss": avg_loss,
            },
            ckpt_path,
        )
        print(f"Saved deploy checkpoint to {ckpt_path}")

    print("Latency-aware fine-tuning completed.")


if __name__ == "__main__":
    main()
