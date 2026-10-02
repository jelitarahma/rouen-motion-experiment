"""
Automated unit tests for RouenDataset.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from rouen_motion.dataset import RouenDataset, natural_sort_key


def test_natural_sort_key():
    p1 = Path("frame_2.jpg")
    p2 = Path("frame_10.jpg")
    p3 = Path("frame_1.jpg")
    sorted_paths = sorted([p1, p2, p3], key=natural_sort_key)
    assert [p.name for p in sorted_paths] == ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"]


def test_invalid_path_raises_error():
    with pytest.raises(FileNotFoundError):
        RouenDataset("non_existent_directory_xyz")


def test_empty_directory_raises_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="No supported image files found"):
            RouenDataset(tmp_dir)


def test_dataset_discovery_and_indexing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir)
        # Create 5 dummy image files
        for i in range(1, 6):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(p / f"frame_{i:04d}.jpg"), img)

        ds = RouenDataset(p, start_frame=0, end_frame=4, step=1)
        assert len(ds) == 4
        frame, path = ds[0]
        assert frame.shape == (50, 50, 3)
        assert path.name == "frame_0001.jpg"

        # Out of bounds
        with pytest.raises(IndexError):
            _ = ds[10]


def test_dataset_step_slicing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir)
        for i in range(1, 7):
            img = np.zeros((20, 20, 3), dtype=np.uint8)
            cv2.imwrite(str(p / f"{i:08d}.jpg"), img)

        ds = RouenDataset(p, start_frame=0, end_frame=6, step=2)
        assert len(ds) == 3
        assert ds[0][1].name == "00000001.jpg"
        assert ds[1][1].name == "00000003.jpg"
        assert ds[2][1].name == "00000005.jpg"
