"""Runner eksperimen tracking (2-frame vs multi-frame)."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# path src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rouen_motion.config import load_config
from rouen_motion.pipeline import RouenMotionPipeline
from rouen_motion.tracking.tracker import TwoFrameTracker, MultiFrameTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rouen motion tracking experiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames to process")
    parser.add_argument("--compare-twoframe", action="store_true", help="Compare 2-frame vs multi-frame tracker")
    return parser.parse_args()


def run_tracker_comparison(pipeline: RouenMotionPipeline, num_frames: int) -> None:
    """Komparasi 2-frame vs multi-frame tracker."""
    print("\n--- Comparing 2-Frame Tracker vs Multi-Frame Tracker (Occlusion Robustness) ---")

    # 2-frame tracker
    pipeline.segmenter.reset()
    two_frame_tracker = TwoFrameTracker(pipeline.config.tracking)
    two_frame_ids = set()

    for idx in range(num_frames):
        raw_frame, _ = pipeline.dataset[idx]
        smoothed_bgr, _ = pipeline.preprocessor.process(raw_frame)
        seg_res = pipeline.segmenter.segment(smoothed_bgr)
        tracks = two_frame_tracker.update(seg_res.regions, frame_idx=idx)
        for t in tracks:
            two_frame_ids.add(t.track_id)

    # multi-frame tracker
    pipeline.segmenter.reset()
    multi_frame_tracker = MultiFrameTracker(pipeline.config.tracking)
    multi_frame_ids = set()

    for idx in range(num_frames):
        raw_frame, _ = pipeline.dataset[idx]
        smoothed_bgr, _ = pipeline.preprocessor.process(raw_frame)
        seg_res = pipeline.segmenter.segment(smoothed_bgr)
        tracks = multi_frame_tracker.update(seg_res.regions, frame_idx=idx)
        for t in tracks:
            multi_frame_ids.add(t.track_id)

    print("\nComparison Results:")
    print(f"  Frames evaluated: {num_frames}")
    print(f"  2-Frame Tracker Total Unique IDs spawned: {len(two_frame_ids)} (Fragmented due to drops)")
    print(f"  Multi-Frame Tracker Total Unique IDs:     {len(multi_frame_ids)} (Maintained across occlusion)")
    print(f"  Multi-Frame Occlusion Recoveries:         {multi_frame_tracker.total_occlusion_recoveries}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.frames is not None:
        config.dataset.end_frame = args.frames
    pipeline = RouenMotionPipeline(config)


    print(f"Loaded dataset: {pipeline.dataset.frame_dir} ({len(pipeline.dataset)} frames configured)")
    print(f"Tracking method: {config.tracking.method.upper()}")
    print("Running tracking experiment...")

    results = pipeline.run_tracking_experiment(max_frames=args.frames)

    print("\nTracking Experiment Completed:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.compare_twoframe:
        frames_to_run = args.frames or min(len(pipeline.dataset), 60)
        run_tracker_comparison(pipeline, frames_to_run)

    print(f"Visualizations saved to: {Path(config.output.directory) / 'tracking'}")


if __name__ == "__main__":
    main()
