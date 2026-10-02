"""Runner eksperimen optical flow (Farneback + multi-frame consistency)."""

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
from rouen_motion.optical_flow.flow import FarnebackOpticalFlow, MultiFrameOpticalFlow, flow_to_hsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rouen optical flow experiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames to process")
    parser.add_argument("--compare-multiframe", action="store_true", help="Compare 2-frame vs multi-frame optical flow")
    return parser.parse_args()


def run_multiframe_comparison(pipeline: RouenMotionPipeline, frame_idx: int = 15) -> None:
    """Komparasi optical flow 2-frame vs multi-frame."""
    print("\n--- Comparing 2-Frame vs Multi-Frame Optical Flow ---")
    out_dir = Path(pipeline.config.output.directory) / "optical_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    two_frame_flow = FarnebackOpticalFlow(pipeline.config.optical_flow.farneback)
    multi_frame_flow = MultiFrameOpticalFlow(
        base_flow=FarnebackOpticalFlow(pipeline.config.optical_flow.farneback),
        config=pipeline.config.optical_flow.multiframe,
    )

    # warmup buffer multi-frame
    start_i = max(0, frame_idx - 3)
    for i in range(start_i, frame_idx):
        f0_raw, _ = pipeline.dataset[i]
        f1_raw, _ = pipeline.dataset[i + 1]
        _, g0 = pipeline.preprocessor.process(f0_raw)
        _, g1 = pipeline.preprocessor.process(f1_raw)
        _ = multi_frame_flow.compute(g0, g1)

    # hitung flow di target frame
    f_prev_raw, _ = pipeline.dataset[frame_idx]
    f_curr_raw, _ = pipeline.dataset[frame_idx + 1]
    curr_bgr, g_prev = pipeline.preprocessor.process(f_prev_raw)
    _, g_curr = pipeline.preprocessor.process(f_curr_raw)

    res_2frame = two_frame_flow.compute(g_prev, g_curr)
    res_mframe = multi_frame_flow.compute(g_prev, g_curr)

    # plot visualisasi
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(flow_to_hsv(res_2frame.flow), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"2-Frame Flow (HSV) - Frame {frame_idx:04d}->{frame_idx+1:04d}", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(flow_to_hsv(res_mframe.flow), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Multi-Frame Flow (HSV)", fontsize=11)
    axes[0, 1].axis("off")

    if res_mframe.forward_backward_error is not None:
        im_err = axes[1, 0].imshow(res_mframe.forward_backward_error, cmap="magma", vmax=5.0)
        axes[1, 0].set_title("Forward-Backward Consistency Error (|F + B_warp|)", fontsize=11)
        axes[1, 0].axis("off")
        fig.colorbar(im_err, ax=axes[1, 0], fraction=0.046, pad=0.04)

    if res_mframe.occlusion_mask is not None:
        axes[1, 1].imshow(res_mframe.occlusion_mask, cmap="Reds")
        occl_pct = float(np.mean(res_mframe.occlusion_mask) * 100.0)
        axes[1, 1].set_title(f"Detected Occlusion / Disocclusion Mask ({occl_pct:.1f}% px)", fontsize=11)
        axes[1, 1].axis("off")

    plt.tight_layout()
    comp_path = out_dir / f"comparison_2frame_vs_multiframe_{frame_idx:04d}.png"
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Optical Flow comparison to: {comp_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.frames is not None:
        config.dataset.end_frame = args.frames
    pipeline = RouenMotionPipeline(config)


    print(f"Loaded dataset: {pipeline.dataset.frame_dir} ({len(pipeline.dataset)} frames configured)")
    print(f"Optical Flow Method: {config.optical_flow.method.upper()}")
    print("Running optical flow experiment...")

    results = pipeline.run_optical_flow_experiment(max_frames=args.frames)

    print("\nOptical Flow Experiment Completed:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.compare_multiframe:
        run_multiframe_comparison(pipeline)

    print(f"Visualizations saved to: {Path(config.output.directory) / 'optical_flow'}")


if __name__ == "__main__":
    main()
