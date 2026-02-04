import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mot17_dataset import MOT17SurveilDataset, collate_fn_mot
from models.objtracknet import ObjTrackNet
from losses.detection_loss import DetectionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 MOT17 fine-tuning for ObjTrackNet")
    parser.add_argument("--mot-root", type=str, required=True, help="Path to MOT17 root")
    parser.add_argument("--coco-ckpt", type=str, required=True, help="Path to Stage 1 COCO checkpoint (.pth)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints_mot")
    return parser.parse_args()


def build_dataloaders(mot_root: str, img_size: int, batch_size: int, num_workers: int):
    train_dataset = MOT17SurveilDataset(
        root=mot_root,
        split="train",
        train=True,
        img_size=img_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_mot,
    )
    return train_loader, train_dataset


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: str,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train MOT]", ncols=120)
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        losses: Dict[str, torch.Tensor] = criterion(outputs, targets)
        loss = losses["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{losses['loss_cls'].item():.4f}",
            "obj": f"{losses['loss_obj'].item():.4f}",
            "box": f"{losses['loss_box'].item():.4f}",
        })

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, train_dataset = build_dataloaders(args.mot_root, args.img_size, args.batch_size, args.num_workers)

    # For MOT we only need 'person' + background, but we'll keep same ObjTrackNet class:
    num_classes = 2  # background + person
    model = ObjTrackNet(num_classes=num_classes, num_human_classes=1, base_channels=32)
    model.to(device)

    # Load COCO pretrain if provided (state dict only)
    if os.path.isfile(args.coco_ckpt):
        ckpt = torch.load(args.coco_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded COCO checkpoint. Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print(f"Warning: COCO checkpoint not found at {args.coco_ckpt}. Training from scratch.")

    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        ckpt_path = Path(args.out_dir) / f"objtracknet_mot_epoch{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
            },
            ckpt_path,
        )
        print(f"Saved MOT checkpoint to {ckpt_path}")

    best_path = Path(args.out_dir) / "objtracknet_mot_last.pth"
    torch.save(model.state_dict(), best_path)
    print(f"Saved final MOT fine-tuned weights to {best_path}")


if __name__ == "__main__":
    main()
