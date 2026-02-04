import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils.transforms import get_train_transforms, get_val_transforms


COCO_PERSON_ID = 1  # standard COCO category ID for 'person'


class CocoSurveilDataset(Dataset):
    """
    COCO 2017 dataset wrapper for SurveilSmartAI / ObjTrackNet.

    It:
    - Loads images and annotations.
    - Converts boxes to [x1,y1,x2,y2].
    - Produces labels and a 'is_human' mask for dual heads.
    """

    def __init__(
        self,
        root: str,
        split: str = "train2017",
        ann_file: str = None,
        train: bool = True,
        img_size: int = 640,
    ):
        assert split in ["train2017", "val2017"]
        if ann_file is None:
            ann_file = os.path.join(root, "annotations", f"instances_{split}.json")

        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split)
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.train = train
        self.transforms = get_train_transforms(img_size) if train else get_val_transforms(img_size)

        # map category id to contiguous label
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}  # 0 reserved for background
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.ids[index]
        info = self.coco.loadImgs(img_id)[0]
        path = info["file_name"]
        img_path = os.path.join(self.img_dir, path)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        is_human = []

        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            cat_id = ann["category_id"]
            label = self.cat_id_to_label.get(cat_id, None)
            if label is None:
                continue
            labels.append(label)
            is_human.append(cat_id == COCO_PERSON_ID)

        if len(boxes) == 0:
            # dummy background object to avoid empty targets
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
            is_human = np.array([False], dtype=bool)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            is_human = np.array(is_human, dtype=bool)

        target: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "is_human": torch.from_numpy(is_human),
            "image_id": torch.tensor([img_id]),
        }

        image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    return images, targets
