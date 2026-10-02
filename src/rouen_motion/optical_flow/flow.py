"""Optical flow: Farneback, consistency check, dan multi-frame smoothing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
import cv2
import numpy as np
from rouen_motion.config import FarnebackConfig, MultiFrameFlowConfig, OpticalFlowConfig


@dataclass
class FlowResult:
    flow: np.ndarray             # (H, W, 2)
    magnitude: np.ndarray        # (H, W)
    angle: np.ndarray            # (H, W) derajat [0, 360)
    occlusion_mask: Optional[np.ndarray] = None  # (H, W) bool
    backward_flow: Optional[np.ndarray] = None
    forward_backward_error: Optional[np.ndarray] = None


class BaseOpticalFlow(ABC):
    @abstractmethod
    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FlowResult:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class FarnebackOpticalFlow(BaseOpticalFlow):
    """Dense optical flow Farneback."""

    def __init__(self, config: Optional[FarnebackConfig] = None) -> None:
        self.config = config or FarnebackConfig()

    def reset(self) -> None:
        pass

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FlowResult:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=self.config.pyr_scale,
            levels=self.config.levels,
            winsize=self.config.winsize,
            iterations=self.config.iterations,
            poly_n=self.config.poly_n,
            poly_sigma=self.config.poly_sigma,
            flags=self.config.flags,
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        return FlowResult(flow=flow, magnitude=magnitude, angle=angle)


class MultiFrameOpticalFlow(BaseOpticalFlow):
    """Optical flow multi-frame (consistency check + temporal smoothing)."""

    def __init__(
        self,
        base_flow: Optional[BaseOpticalFlow] = None,
        config: Optional[MultiFrameFlowConfig] = None,
    ) -> None:
        self.base_flow = base_flow or FarnebackOpticalFlow()
        self.config = config or MultiFrameFlowConfig()
        self.flow_history: Deque[np.ndarray] = deque(maxlen=self.config.temporal_window)

    def reset(self) -> None:
        self.flow_history.clear()
        self.base_flow.reset()

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FlowResult:
        # forward flow
        forward_res = self.base_flow.compute(prev_gray, curr_gray)
        f_flow = forward_res.flow

        # backward flow
        backward_res = self.base_flow.compute(curr_gray, prev_gray)
        b_flow = backward_res.flow

        # forward-backward consistency check
        h, w = prev_gray.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + f_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + f_flow[..., 1]).astype(np.float32)

        # warp backward flow
        b_warped_x = cv2.remap(b_flow[..., 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b_warped_y = cv2.remap(b_flow[..., 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b_warped = np.stack([b_warped_x, b_warped_y], axis=-1)

        # hitung error konsistensi
        fb_error = np.linalg.norm(f_flow + b_warped, axis=-1)
        occlusion_mask = fb_error > self.config.forward_backward_threshold

        # temporal smoothing
        self.flow_history.append(f_flow)
        if self.config.temporal_smoothing and len(self.flow_history) >= 2:
            # moving average
            weights = np.linspace(0.5, 1.0, len(self.flow_history))
            weights /= weights.sum()
            smoothed_flow = np.zeros_like(f_flow)
            for w_val, past_f in zip(weights, self.flow_history):
                smoothed_flow += w_val * past_f
            out_flow = smoothed_flow
        else:
            out_flow = f_flow

        magnitude, angle = cv2.cartToPolar(out_flow[..., 0], out_flow[..., 1], angleInDegrees=True)

        return FlowResult(
            flow=out_flow,
            magnitude=magnitude,
            angle=angle,
            occlusion_mask=occlusion_mask,
            backward_flow=b_flow,
            forward_backward_error=fb_error,
        )


def flow_to_hsv(flow: np.ndarray, max_speed: Optional[float] = None) -> np.ndarray:
    """Visualisasi flow ke format HSV (Hue=arah, Value=kecepatan)."""
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2.0).astype(np.uint8)
    hsv[..., 1] = 255

    if max_speed is None:
        max_speed = max(1.0, float(np.percentile(magnitude, 99)))

    normalized_mag = np.clip((magnitude / max_speed) * 255.0, 0, 255).astype(np.uint8)
    hsv[..., 2] = normalized_mag

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def warp_image_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp gambar pakai flow field (backward warping)."""
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
