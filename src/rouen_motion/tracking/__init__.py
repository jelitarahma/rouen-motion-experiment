"""
Tracking module exports.
"""

from rouen_motion.tracking.tracker import (
    BaseTracker,
    TwoFrameTracker,
    MultiFrameTracker,
    Track,
    TrackState,
    compute_iou,
)
from rouen_motion.config import TrackingConfig


def create_tracker(config: TrackingConfig) -> BaseTracker:
    """Factory function for creating trackers."""
    method = config.method.lower()
    if method in {"multiframe", "multi_frame", "multi"}:
        return MultiFrameTracker(config)
    elif method in {"twoframe", "two_frame", "simple"}:
        return TwoFrameTracker(config)
    else:
        raise ValueError(f"Unsupported tracking method: '{config.method}'")


__all__ = [
    "BaseTracker",
    "TwoFrameTracker",
    "MultiFrameTracker",
    "Track",
    "TrackState",
    "compute_iou",
    "create_tracker",
]
