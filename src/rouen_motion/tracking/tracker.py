"""Tracking modul: baseline 2-frame vs multi-frame tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from rouen_motion.config import TrackingConfig
from rouen_motion.segmentation.base import DetectedRegion


def compute_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """Hitung IoU antara dua bounding box (x, y, w, h)."""
    xa1, ya1, wa, ha = bbox_a
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = bbox_b
    xb2, yb2 = xb1 + wb, yb1 + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


class TrackState(Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    OCCLUDED = "OCCLUDED"
    TERMINATED = "TERMINATED"


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    state: TrackState = TrackState.NEW
    hits: int = 1
    lost_count: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    occlusion_recoveries: int = 0

    def predict(self, damping: float = 0.95) -> Tuple[int, int, int, int]:
        """Prediksi posisi bbox selanjutnya pakai velocity."""
        vx, vy = self.velocity
        new_cx = self.centroid[0] + vx
        new_cy = self.centroid[1] + vy
        x, y, w, h = self.bbox
        pred_x = int(round(new_cx - w / 2.0))
        pred_y = int(round(new_cy - h / 2.0))
        return (pred_x, pred_y, w, h)

    def update(self, region: DetectedRegion, damping: float = 0.95) -> None:
        """Update track dengan deteksi baru."""
        old_cx, old_cy = self.centroid
        new_cx, new_cy = region.centroid

        # update velocity
        inst_vx = new_cx - old_cx
        inst_vy = new_cy - old_cy
        self.velocity = (
            self.velocity[0] * (1.0 - damping) + inst_vx * damping,
            self.velocity[1] * (1.0 - damping) + inst_vy * damping,
        )

        if self.state == TrackState.OCCLUDED:
            self.occlusion_recoveries += 1

        self.bbox = region.bbox
        self.centroid = region.centroid
        self.area = region.area
        self.trajectory.append(region.centroid)
        self.hits += 1
        self.lost_count = 0
        self.state = TrackState.ACTIVE

    def mark_missed(self) -> None:
        """Tandai track hilang/oklusi di frame ini."""
        self.lost_count += 1
        self.state = TrackState.OCCLUDED
        # extrapolasi posisi
        vx, vy = self.velocity
        self.centroid = (self.centroid[0] + vx, self.centroid[1] + vy)
        x, y, w, h = self.bbox
        self.bbox = (int(x + vx), int(y + vy), w, h)
        self.trajectory.append(self.centroid)


class BaseTracker:
    def update(self, detections: List[DetectedRegion], frame_idx: int) -> List[Track]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class TwoFrameTracker(BaseTracker):
    """Tracker 2-frame baseline (hanya matching frame t dan t-1 via IoU)."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self.next_track_id = 1
        self.prev_tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.prev_tracks = {}

    def update(self, detections: List[DetectedRegion], frame_idx: int) -> List[Track]:
        current_tracks: Dict[int, Track] = {}
        unmatched_dets = set(range(len(detections)))

        if self.prev_tracks:
            # matching deteksi dengan track sebelumnya
            for track_id, prev_track in self.prev_tracks.items():
                best_iou = 0.0
                best_det_idx = -1

                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    iou = compute_iou(prev_track.bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx

                if best_iou >= self.config.iou_threshold and best_det_idx != -1:
                    matched_det = detections[best_det_idx]
                    prev_track.update(matched_det)
                    current_tracks[track_id] = prev_track
                    unmatched_dets.remove(best_det_idx)

        # buat track baru untuk deteksi yang unmatched
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self.next_track_id,
                bbox=det.bbox,
                centroid=det.centroid,
                area=det.area,
                trajectory=[det.centroid],
                state=TrackState.ACTIVE,
                hits=1,
            )
            current_tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        self.prev_tracks = current_tracks
        return list(current_tracks.values())


class MultiFrameTracker(BaseTracker):
    """Multi-frame tracker dengan memory oklusi dan estimasi pergerakan."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self.next_track_id = 1
        self.tracks: List[Track] = []
        self.total_occlusion_recoveries = 0

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks = []
        self.total_occlusion_recoveries = 0

    def update(self, detections: List[DetectedRegion], frame_idx: int) -> List[Track]:
        # prediksi posisi track
        predictions = [t.predict(damping=self.config.velocity_damping) for t in self.tracks]

        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        # matching IoU
        matches: List[Tuple[int, int]] = []
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                pred_bbox = predictions[t_idx]
                for d_idx, det in enumerate(detections):
                    iou = compute_iou(pred_bbox, det.bbox)
                    iou_matrix[t_idx, d_idx] = iou

            # greedy match
            while True:
                max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_val < self.config.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matches.append((int(t_idx), int(d_idx)))
                unmatched_tracks.discard(int(t_idx))
                unmatched_dets.discard(int(d_idx))
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # update matched track
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            was_occluded = track.state == TrackState.OCCLUDED
            track.update(detections[d_idx], damping=self.config.velocity_damping)
            if was_occluded:
                self.total_occlusion_recoveries += 1

        # track missed / oklusi
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.mark_missed()

        # hapus track yang lost > max_lost_frames
        self.tracks = [t for t in self.tracks if t.lost_count <= self.config.max_lost_frames]

        # init new track
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_track_id,
                bbox=det.bbox,
                centroid=det.centroid,
                area=det.area,
                trajectory=[det.centroid],
                state=TrackState.NEW if self.config.min_hits > 1 else TrackState.ACTIVE,
                hits=1,
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        visible_tracks = [
            t for t in self.tracks
            if (t.hits >= self.config.min_hits and t.state != TrackState.TERMINATED)
        ]
        return visible_tracks
