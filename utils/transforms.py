import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # image: H x W x C (numpy, BGR from cv2) -> tensor C x H x W (RGB)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target


class Resize:
    def __init__(self, size: Tuple[int, int] = (640, 640)):
        self.size = size

    def __call__(self, image, target):
        h, w = image.shape[-2], image.shape[-1]
        new_h, new_w = self.size
        image = F.resize(image, self.size)
        if "boxes" in target:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / h)
            target["boxes"] = boxes
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            w = image.shape[-1]
            image = F.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def get_train_transforms(img_size: int = 640):
    return Compose([
        ToTensor(),
        Resize((img_size, img_size)),
        RandomHorizontalFlip(0.5),
    ])


def get_val_transforms(img_size: int = 640):
    return Compose([
        ToTensor(),
        Resize((img_size, img_size)),
    ])
