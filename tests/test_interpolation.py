"""
Automated unit tests for motion-compensated frame interpolation.
"""

import numpy as np
from rouen_motion.config import InterpolationConfig
from rouen_motion.interpolation.interpolator import (
    MotionCompensatedInterpolator,
    compute_psnr,
    compute_ssim,
    compute_mae,
)


def test_metrics_identical_images():
    img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    psnr = compute_psnr(img, img)
    ssim = compute_ssim(img, img)
    mae = compute_mae(img, img)

    assert psnr >= 99.0
    assert abs(ssim - 1.0) < 1e-4
    assert mae == 0.0


def test_metrics_perturbed_images():
    img1 = np.ones((80, 80, 3), dtype=np.uint8) * 100
    img2 = img1.copy()
    img2[20:40, 20:40, :] = 120

    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    mae = compute_mae(img1, img2)

    assert 20.0 < psnr < 60.0
    assert 0.8 < ssim < 1.0
    assert mae > 0.0


def test_interpolator_synthesis_shape():
    cfg = InterpolationConfig(alpha=0.5)
    interpolator = MotionCompensatedInterpolator(config=cfg)

    f0 = np.ones((60, 60, 3), dtype=np.uint8) * 50
    f1 = np.ones((60, 60, 3), dtype=np.uint8) * 150
    gt = np.ones((60, 60, 3), dtype=np.uint8) * 100

    res = interpolator.interpolate(f0, f1, alpha=0.5, ground_truth=gt)

    assert res.interpolated_frame.shape == (60, 60, 3)
    assert res.linear_blend_frame.shape == (60, 60, 3)
    assert res.metrics is not None
    assert "motion_psnr_db" in res.metrics
    assert "linear_psnr_db" in res.metrics
    assert res.error_map is not None
    assert res.error_map.shape == (60, 60)
