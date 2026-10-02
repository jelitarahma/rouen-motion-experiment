"""Runner eksperimen segmentasi (MoG2 vs K-Means)."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# path src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rouen_motion.config import load_config
from rouen_motion.pipeline import RouenMotionPipeline
from rouen_motion.segmentation.kmeans import KMeansSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rouen region segmentation experiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames to process")
    parser.add_argument("--compare-kmeans", action="store_true", help="Run comparative analysis against K-Means")
    return parser.parse_args()


def run_kmeans_comparison(pipeline: RouenMotionPipeline, num_frames: int = 10) -> None:
    """Komparasi MoG2 vs K-Means pada frame sampel."""
    print("\n--- Running Segmentation Comparison: MoG2 vs K-Means ---")
    km = KMeansSegmenter()
    out_dir = Path(pipeline.config.output.directory) / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pilih sampel frame
    idx = min(30, len(pipeline.dataset) - 1)
    raw_frame, _ = pipeline.dataset[idx]
    smoothed_bgr, _ = pipeline.preprocessor.process(raw_frame)

    # segmentasi MoG
    mog_res = pipeline.segmenter.segment(smoothed_bgr)

    # segmentasi KMeans
    km_res = km.segment(smoothed_bgr)

    # plot perbandingan
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original Frame ({idx:04d})", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mog_res.mask, cmap="gray")
    axes[0, 1].set_title(f"MoG2 Foreground ({len(mog_res.regions)} objects)", fontsize=12)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(km_res.labeled_image, cmap="tab10")
    axes[1, 0].set_title("K-Means (K=4)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(km_res.mask, cmap="gray")
    axes[1, 1].set_title("K-Means Foreground Mask", fontsize=12)
    axes[1, 1].axis("off")

    plt.tight_layout()
    comp_path = out_dir / "comparison_mog_vs_kmeans.png"
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MoG vs K-Means comparison figure to: {comp_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.frames is not None:
        config.dataset.end_frame = args.frames
    pipeline = RouenMotionPipeline(config)


    print(f"Loaded dataset: {pipeline.dataset.frame_dir} ({len(pipeline.dataset)} frames configured)")
    print(f"Algorithm: {config.segmentation.algorithm.upper()}")
    print("Running segmentation experiment...")

    results = pipeline.run_segmentation_experiment(max_frames=args.frames)

    print("\nSegmentation Experiment Completed:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.compare_kmeans:
        run_kmeans_comparison(pipeline)

    print(f"Visualizations saved to: {Path(config.output.directory) / 'segmentation'}")


if __name__ == "__main__":
    main()
