"""
Interpolation module exports.
"""

from rouen_motion.interpolation.interpolator import (
    MotionCompensatedInterpolator,
    InterpolationResult,
    compute_psnr,
    compute_ssim,
    compute_mae,
)
from rouen_motion.config import InterpolationConfig


def create_interpolator(config: InterpolationConfig) -> MotionCompensatedInterpolator:
    return MotionCompensatedInterpolator(config=config)


__all__ = [
    "MotionCompensatedInterpolator",
    "InterpolationResult",
    "compute_psnr",
    "compute_ssim",
    "compute_mae",
    "create_interpolator",
]
