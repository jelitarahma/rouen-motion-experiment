"""Loader dataset frame Rouen."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np


def natural_sort_key(path: Path) -> Tuple[int | str, ...]:
    """Helper natural sort urutan file frame."""
    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name))


class RouenDataset:
    """Dataset frame video Rouen."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        frame_dir: str | Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> None:
        self.frame_dir = Path(frame_dir)
        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frame directory does not exist: {self.frame_dir}")
        if not self.frame_dir.is_dir():
            raise NotADirectoryError(f"Specified frame path is not a directory: {self.frame_dir}")

        all_files = [
            p for p in self.frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not all_files:
            raise ValueError(
                f"No supported image files found in {self.frame_dir}. "
                f"Expected one of extensions: {self.SUPPORTED_EXTENSIONS}"
            )

        # sort urutan frame
        all_files.sort(key=natural_sort_key)
        self.all_frame_paths = all_files

        total_frames = len(self.all_frame_paths)
        if start_frame < 0 or start_frame >= total_frames:
            raise IndexError(
                f"start_frame={start_frame} out of range for dataset with {total_frames} frames"
            )

        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)

        if end_frame <= start_frame:
            raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})")

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        self.frame_paths: List[Path] = self.all_frame_paths[start_frame:end_frame:step]
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Path]:
        """
        Retrieves frame image (BGR) and its file path by index.
        """
        if idx < 0 or idx >= len(self.frame_paths):
            raise IndexError(f"Index {idx} out of range for sequence of length {len(self.frame_paths)}")
        path = self.frame_paths[idx]
        frame = cv2.imread(str(path))
        if frame is None:
            raise IOError(f"Failed to read image at {path}")
        return frame, path

    def load_frame(self, idx: int) -> np.ndarray:
        """Helper to get only the frame image array."""
        frame, _ = self[idx]
        return frame

    def get_path(self, idx: int) -> Path:
        """Helper to get only the frame path."""
        return self.frame_paths[idx]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Path]]:
        for i in range(len(self)):
            yield self[i]
