"""
Automated unit tests for FramePreprocessor.
"""

import numpy as np
from rouen_motion.config import PreprocessingConfig
from rouen_motion.preprocessing import FramePreprocessor


def test_resize():
    cfg = PreprocessingConfig(resize_width=320, resize_height=180)
    prep = FramePreprocessor(cfg)
    img = np.zeros((576, 1024, 3), dtype=np.uint8)
    resized = prep.resize(img)
    assert resized.shape == (180, 320, 3)


def test_color_conversions():
    prep = FramePreprocessor()
    img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 255  # Blue

    gray = prep.to_gray(img_bgr)
    assert gray.shape == (100, 100)
    assert len(gray.shape) == 2

    rgb = prep.to_rgb(img_bgr)
    assert rgb.shape == (100, 100, 3)
    # Check channel swap
    assert rgb[0, 0, 2] == 255


def test_process_pipeline():
    cfg = PreprocessingConfig(resize_width=200, resize_height=100, apply_gaussian_blur=True)
    prep = FramePreprocessor(cfg)
    raw = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    smoothed, gray = prep.process(raw)
    assert smoothed.shape == (100, 200, 3)
    assert gray.shape == (100, 200)
    assert smoothed.dtype == np.uint8
    assert gray.dtype == np.uint8
