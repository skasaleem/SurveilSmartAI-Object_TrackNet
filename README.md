# SurveilSmartAI / ObjTrackNet (Skeleton Implementation)

This repository contains a **reference PyTorch implementation skeleton** for the SurveilSmartAI / ObjTrackNet framework,
aligned with a three-stage training methodology:

1. Stage 1 – COCO-based object & human detection pretraining.
2. Stage 2 – MOT-based tracking-aware fine-tuning (skeleton hooks only).
3. Stage 3 – Deployment / latency-aware optimisation (placeholders to extend).

The code is written to be **readable and easily extendable**, not as an ultra-optimised production system.

## Features

- Hybrid backbone (Conv + lightweight Transformer) that outputs multi-scale feature maps.
- BiFPN feature fusion.
- Dual detection heads (object + human).
- COCO dataset loader with basic augmentations.
- Training script for Stage 1 (pretraining on COCO 2017).
- Modular structure for adding MOT / CAVIAR datasets and real-time pipeline later.

## Requirements

Install dependencies (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

Key packages:
- torch
- torchvision
- opencv-python
- numpy
- pyyaml
- tqdm
- pycocotools

## Dataset Layout

Point the COCO 2017 dataset to a root folder like:

```text
/path/to/coco2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

Then run training with:

```bash
python train_coco.py --data-root /path/to/coco2017 --epochs 50
```

The script will save checkpoints to `checkpoints/`.


## Training Stages Summary

1. **Stage 1 – COCO Pretraining**
   ```bash
   python train_coco.py --data-root /path/to/coco2017
   ```

2. **Stage 2 – MOT17 Fine-Tuning**
   ```bash
   python train_mot.py --mot-root /path/to/MOT17 --coco-ckpt checkpoints/objtracknet_coco_best.pth
   ```

3. **Stage 3 – Latency-Aware Deployment Fine-Tuning**
   ```bash
   python -m training.train_deploy_latency --data-root /path/to/frames --ckpt checkpoints_mot/objtracknet_mot_last.pth
   ```

## Real-Time Demo with Tracking

Run the real-time(ish) tracking pipeline (webcam by default):

```bash
python -m runtime.realtime_pipeline --ckpt checkpoints_mot/objtracknet_mot_last.pth
```

This will:
- Run ObjTrackNet for detection.
- Use an IoU-based online tracker to assign IDs.
- Adaptively skip frames if processing exceeds the target FPS.
