"""
Automated unit tests for Segmentation algorithms (MoG and KMeans).
"""

import numpy as np
from rouen_motion.config import MOGConfig, KMeansConfig
from rouen_motion.segmentation.mog import MixtureOfGaussiansSegmenter
from rouen_motion.segmentation.kmeans import KMeansSegmenter


def test_mog_segmentation_shape_and_type():
    cfg = MOGConfig(history=10, min_contour_area=20.0, detect_shadows=False)
    seg = MixtureOfGaussiansSegmenter(cfg)


    # Simulate static background
    bg = np.ones((120, 160, 3), dtype=np.uint8) * 100
    for _ in range(25):
        seg.segment(bg)


    # Frame with a moving foreground square
    fg_frame = bg.copy()
    fg_frame[40:80, 50:90, :] = 255  # bright moving object

    res = seg.segment(fg_frame)

    assert res.mask.shape == (120, 160)
    assert res.mask.dtype == np.uint8
    assert res.labeled_image.shape == (120, 160)
    # Binary mask contains only 0 and 255
    unique_vals = set(np.unique(res.mask))
    assert unique_vals.issubset({0, 255})
    assert len(res.regions) >= 1

    region = res.regions[0]
    assert len(region.bbox) == 4
    assert len(region.centroid) == 2
    assert region.area > 0


def test_kmeans_segmenter():
    cfg = KMeansConfig(n_clusters=3, max_iter=10)
    km = KMeansSegmenter(cfg)
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    res = km.segment(frame)

    assert res.mask.shape == (100, 100)
    assert res.labeled_image.shape == (100, 100)
    assert res.metadata["clusters"] == 3
