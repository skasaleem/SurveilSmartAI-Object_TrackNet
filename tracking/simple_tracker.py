from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1,y1,x2,y2]
    score: float
    age: int = 0
    hits: int = 1
    miss: int = 0


class IoUTracker:
    """
    Minimal IoU-based multi-object tracker.

    - Maintains a list of active tracks.
    - Associates detections to tracks via IoU and greedy matching.
    - Spawns new tracks for unmatched detections.
    - Removes tracks with too many consecutive misses.

    This is intentionally lightweight but fully functional.
    """

    def __init__(self, iou_threshold: float = 0.3, max_miss: int = 30, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        # a, b: [x1,y1,x2,y2]
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    def update(self, detections: List[np.ndarray], scores: List[float]) -> List[Track]:
        """
        detections: list of [x1,y1,x2,y2]
        scores: list of detection scores
        Returns active tracks after update (including newly created).
        """
        detections = list(detections)
        scores = list(scores)

        # Age all tracks
        for t in self.tracks.values():
            t.age += 1
            t.miss += 1

        # Build IoU matrix: tracks x detections
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tb = self.tracks[tid].bbox
                for di, db in enumerate(detections):
                    iou_matrix[ti, di] = self._iou(tb, db)
        else:
            iou_matrix = None

        matched_dets = set()
        matched_tracks = set()

        # Greedy matching by IoU
        if iou_matrix is not None:
            while True:
                ti, di = divmod(iou_matrix.argmax(), iou_matrix.shape[1])
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                tid = track_ids[ti]
                if tid in matched_tracks or di in matched_dets:
                    iou_matrix[ti, di] = -1.0
                    continue
                # match
                self.tracks[tid].bbox = detections[di]
                self.tracks[tid].score = scores[di]
                self.tracks[tid].hits += 1
                self.tracks[tid].miss = 0
                matched_tracks.add(tid)
                matched_dets.add(di)
                iou_matrix[ti, :] = -1.0
                iou_matrix[:, di] = -1.0

        # Unmatched detections -> new tracks
        for di, db in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, bbox=db, score=scores[di])

        # Remove old tracks
        to_remove = [tid for tid, t in self.tracks.items() if t.miss > self.max_miss]
        for tid in to_remove:
            del self.tracks[tid]

        # Return visible tracks (optionally filter by hits)
        visible = [t for t in self.tracks.values() if t.hits >= self.min_hits]
        return visible
