"""Runner eksperimen interpolasi frame (evaluasi vs ground truth)."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# path src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rouen_motion.config import load_config
from rouen_motion.pipeline import RouenMotionPipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rouen frame interpolation experiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames to process")
    parser.add_argument("--alpha", type=float, default=0.5, help="Intermediate time fraction (0.5 for mid-frame)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.interpolation.alpha = args.alpha
    if args.frames is not None:
        config.dataset.end_frame = args.frames

    pipeline = RouenMotionPipeline(config)


    print(f"Loaded dataset: {pipeline.dataset.frame_dir} ({len(pipeline.dataset)} frames configured)")
    print(f"Interpolation Alpha: {config.interpolation.alpha}")
    print("Running frame interpolation experiment with ground-truth verification...")

    results = pipeline.run_interpolation_experiment(max_frames=args.frames)

    print("\nFrame Interpolation Experiment Completed:")
    mc = results.get("motion_compensated", {})
    lb = results.get("linear_blend_baseline", {})
    gain = results.get("relative_gain", {})

    print(f"  Evaluations performed: {results.get('evaluations_count')}")
    print(f"  Motion-Compensated Interpolation:")
    print(f"    PSNR: {mc.get('mean_psnr_db', 0):.2f} dB")
    print(f"    SSIM: {mc.get('mean_ssim', 0):.4f}")
    print(f"    MAE:  {mc.get('mean_mae', 0):.2f} intensity units")
    print(f"  Linear Blending Baseline:")
    print(f"    PSNR: {lb.get('mean_psnr_db', 0):.2f} dB")
    print(f"    SSIM: {lb.get('mean_ssim', 0):.4f}")
    print(f"    MAE:  {lb.get('mean_mae', 0):.2f} intensity units")
    print(f"  Quantitative Gains (Global):")
    print(f"    PSNR Gain: {gain.get('psnr_improvement_db', 0):+.2f} dB")
    print(f"    SSIM Gain: {gain.get('ssim_improvement', 0):+.4f}")

    roi = results.get("moving_vehicles_roi", {})
    if roi:
        mc_roi = roi.get("motion_compensated", {})
        lb_roi = roi.get("linear_blend_baseline", {})
        print(f"\n  Moving Vehicles Region (ROI Analysis):")
        print(f"    Motion-Compensated PSNR: {mc_roi.get('mean_psnr_db', 0):.2f} dB | MAE: {mc_roi.get('mean_mae', 0):.2f}")
        print(f"    Linear Blending PSNR:   {lb_roi.get('mean_psnr_db', 0):.2f} dB | MAE: {lb_roi.get('mean_mae', 0):.2f}")
        print(f"    ROI PSNR Difference:    {roi.get('roi_psnr_difference_db', 0):+.2f} dB")

    print(f"\nVisualizations saved to: {Path(config.output.directory) / 'interpolation'}")



if __name__ == "__main__":
    main()
