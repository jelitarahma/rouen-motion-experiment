"""
Utility script to export Rouen frames and experiment results into playable .mp4 video files.

Usage:
    python -m experiments.export_video --type all
    python -m experiments.export_video --type original
    python -m experiments.export_video --type tracking
    python -m experiments.export_video --type flow
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from rouen_motion.config import load_config
from rouen_motion.dataset import RouenDataset
from rouen_motion.preprocessing import FramePreprocessor
from rouen_motion.segmentation.mog import MixtureOfGaussiansSegmenter
from rouen_motion.tracking.tracker import MultiFrameTracker
from rouen_motion.optical_flow.flow import FarnebackOpticalFlow, flow_to_hsv
from rouen_motion.visualization.visualizer import get_track_color


def export_original_video(dataset: RouenDataset, out_path: Path, max_frames: int = 200, fps: int = 25) -> None:
    """Exports raw video frames into an mp4 video."""
    print(f"Creating original video at: {out_path}...")
    num_frames = min(len(dataset), max_frames)
    first_frame, _ = dataset[0]
    h, w = first_frame.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        frame, _ = dataset[i]
        writer.write(frame)

    writer.release()
    print(f"-> Selesai! Video asli tersimpan di: {out_path}")


def export_tracking_video(dataset: RouenDataset, config, out_path: Path, max_frames: int = 200, fps: int = 25) -> None:
    """Exports motion tracking visualization into an mp4 video."""
    print(f"Creating tracking video at: {out_path}...")
    preprocessor = FramePreprocessor(config.preprocessing)
    segmenter = MixtureOfGaussiansSegmenter(config.segmentation.mog)
    tracker = MultiFrameTracker(config.tracking)

    num_frames = min(len(dataset), max_frames)
    w, h = config.preprocessing.resize_width, config.preprocessing.resize_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        raw_frame, _ = dataset[i]
        bgr, _ = preprocessor.process(raw_frame)
        seg_res = segmenter.segment(bgr)
        tracks = tracker.update(seg_res.regions, frame_idx=i)

        for t in tracks:
            color = get_track_color(t.track_id)
            x, y, bw, bh = t.bbox
            cv2.rectangle(bgr, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                bgr,
                f"ID:{t.track_id}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        writer.write(bgr)

    writer.release()
    print(f"-> Selesai! Video tracking tersimpan di: {out_path}")


def export_flow_video(dataset: RouenDataset, config, out_path: Path, max_frames: int = 150, fps: int = 20) -> None:
    """Exports dense optical flow (HSV color wheel) into an mp4 video."""
    print(f"Creating optical flow video at: {out_path}...")
    preprocessor = FramePreprocessor(config.preprocessing)
    flow_estimator = FarnebackOpticalFlow(config.optical_flow.farneback)

    num_frames = min(len(dataset), max_frames)
    w, h = config.preprocessing.resize_width, config.preprocessing.resize_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    prev_raw, _ = dataset[0]
    _, prev_gray = preprocessor.process(prev_raw)

    for i in range(1, num_frames):
        curr_raw, _ = dataset[i]
        curr_bgr, curr_gray = preprocessor.process(curr_raw)
        res = flow_estimator.compute(prev_gray, curr_gray)
        flow_hsv = flow_to_hsv(res.flow)
        writer.write(flow_hsv)
        prev_gray = curr_gray

    writer.release()
    print(f"-> Selesai! Video optical flow tersimpan di: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment results to .mp4 video")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--type", type=str, default="all", choices=["all", "original", "tracking", "flow"])
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to render into video")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second for output video")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = RouenDataset(config.dataset.frame_dir, start_frame=0, end_frame=args.frames)
    out_dir = Path(config.output.directory)

    if args.type in ("all", "original"):
        export_original_video(dataset, out_dir / "rouen_original.mp4", max_frames=args.frames)

    if args.type in ("all", "tracking"):
        export_tracking_video(dataset, config, out_dir / "rouen_tracking.mp4", max_frames=args.frames)

    if args.type in ("all", "flow"):
        export_flow_video(dataset, config, out_dir / "rouen_flow.mp4", max_frames=min(args.frames, 150))


if __name__ == "__main__":
    main()
