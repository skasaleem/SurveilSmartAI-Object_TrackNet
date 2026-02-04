import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_train_transforms, get_val_transforms


class MOT17SurveilDataset(Dataset):
    """
    Minimal MOT17 dataset loader for frame-wise training.

    Expected layout:

      MOT17_ROOT/
        train/
          MOT17-02-FRCNN/
            img1/
            gt/gt.txt
          ...
        test/
          ...

    This loader uses the 'train' split and reads gt/gt.txt to construct
    frame-level annotations (boxes + track_ids + class=person).

    It returns single frames (no temporal clips) to keep training simple.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train: bool = True,
        img_size: int = 640,
    ):
        assert split in ["train", "test"]
        self.root = root
        self.split = split
        self.train = train
        self.transforms = get_train_transforms(img_size) if train else get_val_transforms(img_size)

        self.sequences = []
        self.frame_index: List[Tuple[str, int]] = []  # (seq_path, frame_id)

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"MOT17 split directory not found: {split_dir}")

        for seq_name in sorted(os.listdir(split_dir)):
            seq_path = os.path.join(split_dir, seq_name)
            if not os.path.isdir(seq_path):
                continue
            img_dir = os.path.join(seq_path, "img1")
            gt_path = os.path.join(seq_path, "gt", "gt.txt")
            if not os.path.isdir(img_dir) or not os.path.isfile(gt_path):
                continue

            # Parse GT
            frame_to_anns: Dict[int, List[Dict[str, Any]]] = {}
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 7:
                        continue
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    conf = float(parts[6])
                    if conf == 0:
                        continue
                    ann = {
                        "track_id": track_id,
                        "bbox": [x, y, w, h],
                    }
                    frame_to_anns.setdefault(frame_id, []).append(ann)

            # Build frame index
            img_files = sorted(os.listdir(img_dir))
            for img_file in img_files:
                if not img_file.lower().endswith((".jpg", ".png")):
                    continue
                frame_id = int(os.path.splitext(img_file)[0])
                if frame_id not in frame_to_anns:
                    # allow images without GT (will be treated as background)
                    frame_to_anns.setdefault(frame_id, [])
                self.frame_index.append((seq_path, frame_id))

            self.sequences.append({"path": seq_path, "frame_to_anns": frame_to_anns})

        # Build lookup for frame_to_anns by (seq_path, frame_id)
        self._frame_lookup: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for seq in self.sequences:
            seq_path = seq["path"]
            fta = seq["frame_to_anns"]
            for frame_id, anns in fta.items():
                self._frame_lookup[(seq_path, frame_id)] = anns

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, index: int):
        seq_path, frame_id = self.frame_index[index]
        img_dir = os.path.join(seq_path, "img1")
        img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        if not os.path.isfile(img_path):
            # try png fallback
            img_path = os.path.join(img_dir, f"{frame_id:06d}.png")
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        anns = self._frame_lookup.get((seq_path, frame_id), [])

        boxes = []
        labels = []
        track_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # 'person' class for MOT
            track_ids.append(ann["track_id"])

        if len(boxes) == 0:
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
            track_ids = np.array([-1], dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            track_ids = np.array(track_ids, dtype=np.int64)

        target: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "track_ids": torch.from_numpy(track_ids),
            "image_id": torch.tensor([frame_id]),
        }

        image, target = self.transforms(image, target)
        return image, target


def collate_fn_mot(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    return images, targets
