"""Preprocessing frame (resize, filtering blur, konversi warna)."""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from rouen_motion.config import PreprocessingConfig


class FramePreprocessor:
    """Preprocessor frame video."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize resolusi frame."""
        target_size = (self.config.resize_width, self.config.resize_height)
        if (image.shape[1], image.shape[0]) == target_size:
            return image
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def blur(self, image: np.ndarray) -> np.ndarray:
        """Filtering gaussian blur untuk reduce noise."""
        if not self.config.apply_gaussian_blur:
            return image
        ksize = tuple(self.config.blur_kernel_size)
        # pastikan kernel ganjil
        kx = ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1
        ky = ksize[1] if ksize[1] % 2 != 0 else ksize[1] + 1
        return cv2.GaussianBlur(image, (kx, ky), self.config.blur_sigma)

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single-channel Grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Converts OpenCV BGR image to standard RGB."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs standard preprocessing pipeline.
        Returns:
            resized_bgr: Resized and smoothed BGR image.
            resized_gray: Resized and smoothed Grayscale image.
        """
        resized = self.resize(image)
        smoothed = self.blur(resized)
        gray = self.to_gray(smoothed)
        return smoothed, gray
