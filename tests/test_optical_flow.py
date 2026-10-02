"""
Automated unit tests for optical flow modules.
"""

import numpy as np
from rouen_motion.config import FarnebackConfig, MultiFrameFlowConfig
from rouen_motion.optical_flow.flow import (
    FarnebackOpticalFlow,
    MultiFrameOpticalFlow,
    flow_to_hsv,
    warp_image_with_flow,
)


def test_farneback_flow_shape_and_magnitude():
    cfg = FarnebackConfig(levels=1, winsize=9, iterations=2)
    estimator = FarnebackOpticalFlow(cfg)

    # Frame 1: Circle at center
    h, w = 80, 80
    f1 = np.zeros((h, w), dtype=np.uint8)
    f1[30:50, 30:50] = 255

    # Frame 2: Circle shifted right by 4 pixels
    f2 = np.zeros((h, w), dtype=np.uint8)
    f2[30:50, 34:54] = 255

    res = estimator.compute(f1, f2)

    assert res.flow.shape == (h, w, 2)
    assert res.magnitude.shape == (h, w)
    assert res.angle.shape == (h, w)
    assert res.flow.dtype == np.float32

    # Flow inside shifted box should point right (positive dx)
    sub_flow_x = res.flow[35:45, 35:45, 0]
    assert np.mean(sub_flow_x) > 0.5


def test_flow_to_hsv_representation():
    flow = np.zeros((50, 50, 2), dtype=np.float32)
    flow[:, :, 0] = 2.0  # Horizontal motion
    hsv_bgr = flow_to_hsv(flow)

    assert hsv_bgr.shape == (50, 50, 3)
    assert hsv_bgr.dtype == np.uint8


def test_multiframe_optical_flow_consistency():
    base = FarnebackOpticalFlow(FarnebackConfig(levels=1, winsize=9, iterations=2))
    cfg = MultiFrameFlowConfig(temporal_window=3, forward_backward_threshold=1.0)
    mf_flow = MultiFrameOpticalFlow(base_flow=base, config=cfg)

    f1 = np.ones((60, 60), dtype=np.uint8) * 50
    f2 = f1.copy()
    f2[20:40, 20:40] = 200

    res = mf_flow.compute(f1, f2)
    assert res.occlusion_mask is not None
    assert res.occlusion_mask.shape == (60, 60)
    assert res.forward_backward_error is not None


def test_image_warping():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = 255

    # Flow shifting backward by 5 pixels in x
    flow = np.zeros((50, 50, 2), dtype=np.float32)
    flow[:, :, 0] = 5.0

    warped = warp_image_with_flow(img, flow)
    assert warped.shape == (50, 50, 3)
