import argparse
import time

import cv2
import numpy as np
import torch

from models.objtracknet import ObjTrackNet
from utils.transforms import ToTensor, Resize
from tracking.simple_tracker import IoUTracker
from inference_demo import decode_predictions  # reuse decoder


def run_realtime(
    source: str,
    ckpt_path: str,
    img_size: int = 640,
    device: str = "cuda",
    target_fps: float = 20.0,
):
    """
    Real-time-style pipeline:
      - Reads frames from webcam/RTSP/video file.
      - Runs ObjTrackNet detection.
      - Tracks objects with IoUTracker.
      - Measures latency; if too slow, adaptively skips frames.
    """
    model = ObjTrackNet(num_classes=10, num_human_classes=1, base_channels=32)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    tracker = IoUTracker(iou_threshold=0.3, max_miss=30, min_hits=1)
    to_tensor = ToTensor()
    resize = Resize((img_size, img_size))

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    # Compute allowed per-frame budget (seconds)
    frame_budget = 1.0 / max(target_fps, 1e-3)
    print(f"Target FPS: {target_fps} -> per-frame budget {frame_budget:.4f}s")

    last_time = time.time()
    frame_idx = 0
    skip_next = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        now = time.time()
        elapsed_since_last = now - last_time

        # Simple adaptive frame skipping
        if elapsed_since_last < frame_budget * 0.5 and skip_next:
            # skip processing, just display
            cv2.imshow("SurveilSmartAI", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        last_time = now

        # Preprocess
        img_tensor, _ = to_tensor(frame, {})
        img_tensor, _ = resize(img_tensor, {})
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            boxes, scores = decode_predictions(
                outputs["obj_cls"],
                outputs["obj_reg"],
                outputs["obj_obj"],
                conf_threshold=0.3,
                obj_threshold=0.3,
                img_size=img_size,
            )
        t1 = time.time()

        # Tracking
        tracks = tracker.update(boxes, scores)

        # Draw detections + track IDs
        vis = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"ID {t.track_id} {t.score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        infer_time = t1 - t0
        total_time = time.time() - now
        fps = 1.0 / max(total_time, 1e-6)

        cv2.putText(
            vis,
            f"FPS: {fps:.1f} (infer {infer_time*1000:.1f} ms)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Decide whether to skip next frame based on budget
        skip_next = total_time > frame_budget

        cv2.imshow("SurveilSmartAI", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time pipeline with tracking and latency-awareness")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--source", type=str, default="webcam", help="webcam, path/to/video.mp4, or RTSP URL")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-fps", type=float, default=20.0)
    args = parser.parse_args()

    run_realtime(
        source=args.source,
        ckpt_path=args.ckpt,
        img_size=args.img_size,
        device=args.device,
        target_fps=args.target_fps,
    )


if __name__ == "__main__":
    main()
