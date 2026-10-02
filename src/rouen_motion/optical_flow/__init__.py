"""
Optical flow module exports.
"""

from rouen_motion.optical_flow.flow import (
    BaseOpticalFlow,
    FarnebackOpticalFlow,
    MultiFrameOpticalFlow,
    FlowResult,
    flow_to_hsv,
    warp_image_with_flow,
)
from rouen_motion.config import OpticalFlowConfig


def create_optical_flow(config: OpticalFlowConfig, multi_frame: bool = False) -> BaseOpticalFlow:
    """Factory function for optical flow estimation."""
    base = FarnebackOpticalFlow(config.farneback)
    if multi_frame:
        return MultiFrameOpticalFlow(base_flow=base, config=config.multiframe)
    return base


__all__ = [
    "BaseOpticalFlow",
    "FarnebackOpticalFlow",
    "MultiFrameOpticalFlow",
    "FlowResult",
    "flow_to_hsv",
    "warp_image_with_flow",
    "create_optical_flow",
]
