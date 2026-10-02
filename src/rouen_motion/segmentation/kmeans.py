"""Baseline segmentasi K-Means."""

from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np
from rouen_motion.config import KMeansConfig
from rouen_motion.segmentation.base import BaseSegmenter, DetectedRegion, SegmentationResult


class KMeansSegmenter(BaseSegmenter):
    """Baseline K-Means (kluster warna)."""

    def __init__(self, config: Optional[KMeansConfig] = None) -> None:
        self.config = config or KMeansConfig()
        self.frame_count = 0

    def reset(self) -> None:
        self.frame_count = 0

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Segmentasi frame dengan K-Means di ruang Lab."""
        self.frame_count += 1
        h, w = frame.shape[:2]

        # konversi ke Lab
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            pixel_features = lab.reshape((-1, 3)).astype(np.float32)
        else:
            pixel_features = frame.reshape((-1, 1)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.config.max_iter,
            self.config.epsilon,
        )

        k = self.config.n_clusters
        _, labels, centers = cv2.kmeans(
            pixel_features,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        labels_img = labels.reshape((h, w)).astype(np.int32)

        # ambil cluster non-dominant sebagai foreground
        counts = np.bincount(labels.flatten())
        dominant_cluster = int(np.argmax(counts))

        # mask cluster
        mask = np.where(labels_img != dominant_cluster, 255, 0).astype(np.uint8)

        # ekstrak region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_regions: List[DetectedRegion] = []
        reg_id = 1
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 100.0 or area > 50000.0:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            moments = cv2.moments(c)
            cx = float(moments["m10"] / moments["m00"]) if moments["m00"] > 0 else float(x + bw / 2.0)
            cy = float(moments["m01"] / moments["m00"]) if moments["m00"] > 0 else float(y + bh / 2.0)

            detected_regions.append(
                DetectedRegion(
                    region_id=reg_id,
                    bbox=(x, y, bw, bh),
                    centroid=(cx, cy),
                    area=area,
                    contour=c,
                )
            )
            reg_id += 1

        metadata = {
            "algorithm": "K-Means (Color Spatial)",
            "clusters": k,
            "dominant_cluster": dominant_cluster,
            "detected_count": len(detected_regions),
        }

        return SegmentationResult(
            mask=mask,
            labeled_image=labels_img,
            regions=detected_regions,
            metadata=metadata,
        )
