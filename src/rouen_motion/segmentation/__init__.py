"""
Segmentation module for video frames.
"""

from rouen_motion.segmentation.base import BaseSegmenter, DetectedRegion, SegmentationResult
from rouen_motion.segmentation.mog import MixtureOfGaussiansSegmenter
from rouen_motion.segmentation.kmeans import KMeansSegmenter
from rouen_motion.config import SegmentationConfig


def create_segmenter(config: SegmentationConfig) -> BaseSegmenter:
    """Factory function to instantiate the chosen segmenter."""
    algo = config.algorithm.lower()
    if algo in {"mog", "mog2", "mixture_of_gaussians", "gmm"}:
        return MixtureOfGaussiansSegmenter(config.mog)
    elif algo in {"kmeans", "k-means"}:
        return KMeansSegmenter(config.kmeans)
    else:
        raise ValueError(
            f"Unsupported segmentation algorithm: '{config.algorithm}'. "
            "Supported algorithms: 'mog', 'kmeans'"
        )


__all__ = [
    "BaseSegmenter",
    "DetectedRegion",
    "SegmentationResult",
    "MixtureOfGaussiansSegmenter",
    "KMeansSegmenter",
    "create_segmenter",
]
