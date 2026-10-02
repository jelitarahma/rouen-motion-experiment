"""
Automated unit tests for motion tracking modules.
"""

import numpy as np
from rouen_motion.config import TrackingConfig
from rouen_motion.segmentation.base import DetectedRegion
from rouen_motion.tracking.tracker import (
    TwoFrameTracker,
    MultiFrameTracker,
    TrackState,
    compute_iou,
)


def test_iou_computation():
    # Exactly identical boxes
    box_a = (10, 10, 20, 20)
    assert compute_iou(box_a, box_a) == 1.0

    # Non-overlapping
    box_b = (100, 100, 20, 20)
    assert compute_iou(box_a, box_b) == 0.0

    # Half overlap: box_a (0, 0, 10, 10) area 100, box_c (5, 0, 10, 10) area 100
    # intersection: 5x10 = 50, union: 100 + 100 - 50 = 150 -> 1/3
    box_1 = (0, 0, 10, 10)
    box_2 = (5, 0, 10, 10)
    assert abs(compute_iou(box_1, box_2) - 1.0 / 3.0) < 1e-4


def test_multiframe_tracker_occlusion_handling():
    cfg = TrackingConfig(iou_threshold=0.2, max_lost_frames=3, min_hits=1)
    tracker = MultiFrameTracker(cfg)

    # Frame 1: Object appears at (100, 100)
    det1 = DetectedRegion(
        region_id=1,
        bbox=(100, 100, 30, 30),
        centroid=(115.0, 115.0),
        area=900.0,
        contour=np.array([]),
    )
    tracks_f1 = tracker.update([det1], frame_idx=0)
    assert len(tracks_f1) == 1
    orig_id = tracks_f1[0].track_id

    # Frame 2: Object moves slightly to (105, 105)
    det2 = DetectedRegion(
        region_id=1,
        bbox=(105, 105, 30, 30),
        centroid=(120.0, 120.0),
        area=900.0,
        contour=np.array([]),
    )
    tracks_f2 = tracker.update([det2], frame_idx=1)
    assert tracks_f2[0].track_id == orig_id
    assert tracks_f2[0].state == TrackState.ACTIVE

    # Frame 3: Object is OCCLUDED (no detection)
    tracks_f3 = tracker.update([], frame_idx=2)
    # The track is marked occluded, kept alive in memory
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.OCCLUDED

    # Frame 4: Object re-emerges at predicted location (115, 115)
    det4 = DetectedRegion(
        region_id=1,
        bbox=(115, 115, 30, 30),
        centroid=(130.0, 130.0),
        area=900.0,
        contour=np.array([]),
    )
    tracks_f4 = tracker.update([det4], frame_idx=3)
    # Should RECOVER same track ID!
    assert len(tracks_f4) == 1
    assert tracks_f4[0].track_id == orig_id
    assert tracker.total_occlusion_recoveries >= 1
