"""Segmentasi MoG2."""

from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np
from rouen_motion.config import MOGConfig
from rouen_motion.segmentation.base import BaseSegmenter, DetectedRegion, SegmentationResult


class MixtureOfGaussiansSegmenter(BaseSegmenter):
    """Segmenter MoG2."""

    def __init__(self, config: Optional[MOGConfig] = None) -> None:
        self.config = config or MOGConfig()
        self.subtractor: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        k_size = tuple(self.config.morph_kernel_size)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
        self.frame_count = 0

    def reset(self) -> None:
        """Reset background model."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self.frame_count = 0

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Segmentasi moving regions."""
        self.frame_count += 1
        learning_rate = self.config.learning_rate

        # background subtraction
        raw_mask = self.subtractor.apply(frame, learningRate=learning_rate)

        # suppress shadow (127)
        if self.config.detect_shadows:
            clean_mask = np.where(raw_mask == 255, 255, 0).astype(np.uint8)
        else:
            clean_mask = raw_mask.copy()

        # filtering...
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # cari kontur
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_regions: List[DetectedRegion] = []
        labeled_mask = np.zeros(clean_mask.shape, dtype=np.int32)
        region_id_counter = 1

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            cv2.drawContours(labeled_mask, [c], -1, region_id_counter, thickness=-1)

            detected_regions.append(
                DetectedRegion(
                    region_id=region_id_counter,
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    area=area,
                    contour=c,
                    confidence=min(1.0, area / 500.0),
                )
            )
            region_id_counter += 1

        metadata = {
            "algorithm": "MixtureOfGaussians (MOG2)",
            "frame_count": self.frame_count,
            "detected_count": len(detected_regions),
            "history": self.config.history,
            "var_threshold": self.config.var_threshold,
        }

        return SegmentationResult(
            mask=clean_mask,
            labeled_image=labeled_mask,
            regions=detected_regions,
            raw_mask=raw_mask,
            metadata=metadata,
        )
