import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_dataset import CocoSurveilDataset, collate_fn
from models.objtracknet import ObjTrackNet
from losses.detection_loss import DetectionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 COCO training for ObjTrackNet")
    parser.add_argument("--data-root", type=str, required=True, help="Path to COCO2017 root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    return parser.parse_args()


def build_dataloaders(data_root: str, img_size: int, batch_size: int, num_workers: int):
    train_dataset = CocoSurveilDataset(
        root=data_root,
        split="train2017",
        train=True,
        img_size=img_size,
    )
    val_dataset = CocoSurveilDataset(
        root=data_root,
        split="val2017",
        train=False,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_dataset


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
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=120)
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


@torch.no_grad()
def validate(model: nn.Module, criterion: nn.Module, loader, device: str, epoch: int):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", ncols=120)
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        losses: Dict[str, torch.Tensor] = criterion(outputs, targets)
        loss = losses["loss_total"]
        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
        })
    return total_loss / len(loader)


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # Build dataloaders
    train_loader, val_loader, train_dataset = build_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    num_classes = len(train_dataset.cat_id_to_label) + 1  # +1 for background (label 0)

    model = ObjTrackNet(num_classes=num_classes, num_human_classes=1, base_channels=32)
    model.to(device)

    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        val_loss = validate(model, criterion, val_loader, device, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        ckpt_path = Path(args.out_dir) / f"objtracknet_coco_epoch{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            ckpt_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(args.out_dir) / "objtracknet_coco_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
