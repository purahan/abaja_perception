"""
Minimal IoU-based tracker with per-track class-vote smoothing.

Why this exists: the rulebook explicitly penalizes "unstable, intermittent,
flickering, delayed, or inconsistent detections" and forbids offline
post-processing — so smoothing has to happen inside the live inference
loop, frame by frame, not after the run. This is intentionally simple
(no Kalman filter, no re-ID) because at 5-25m in a static bay, greedy IoU
association across consecutive frames is more than enough and keeps
latency low on the Orin.
"""

from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Detection:
    class_id: int
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coords


@dataclass
class Track:
    track_id: int
    bbox: tuple
    class_history: deque = field(default_factory=lambda: deque(maxlen=5))
    conf_history: deque = field(default_factory=lambda: deque(maxlen=5))
    hits: int = 0
    misses: int = 0

    def smoothed_class(self) -> int:
        return Counter(self.class_history).most_common(1)[0][0]

    def smoothed_conf(self) -> float:
        return sum(self.conf_history) / len(self.conf_history)


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5,
                 smoothing_window: int = 5, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.smoothing_window = smoothing_window
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        unmatched_dets = list(range(len(detections)))
        for track in self.tracks:
            best_iou, best_idx = 0.0, -1
            for i in unmatched_dets:
                score = iou(track.bbox, detections[i].bbox)
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_iou >= self.iou_threshold:
                det = detections[best_idx]
                track.bbox = det.bbox
                track.class_history.append(det.class_id)
                if len(track.class_history) > self.smoothing_window:
                    track.class_history.popleft()
                track.conf_history.append(det.conf)
                if len(track.conf_history) > self.smoothing_window:
                    track.conf_history.popleft()
                track.hits += 1
                track.misses = 0
                unmatched_dets.remove(best_idx)
            else:
                track.misses += 1

        for i in unmatched_dets:
            det = detections[i]
            t = Track(track_id=self._next_id, bbox=det.bbox)
            t.class_history.append(det.class_id)
            t.conf_history.append(det.conf)
            t.hits = 1
            self._next_id += 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.hits >= self.min_hits]
