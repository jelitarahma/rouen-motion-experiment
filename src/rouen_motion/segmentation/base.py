"""
Base classes and result data structures for region segmentation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np




@dataclass
class DetectedRegion:
    """Represents a segmented moving region / connected component."""
    region_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[float, float]    # (cx, cy)
    area: float
    contour: np.ndarray
    confidence: float = 1.0


@dataclass
class SegmentationResult:
    """Contains segmentation masks, labeled regions, and extracted components."""
    mask: np.ndarray                   # Binary mask (0 or 255)
    labeled_image: np.ndarray          # Color-coded or indexed label map
    regions: List[DetectedRegion] = field(default_factory=list)
    raw_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSegmenter(ABC):
    """Abstract interface for video frame region segmenters."""

    @abstractmethod
    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Processes a single frame and returns segmentation result."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal state / temporal model."""
        pass
