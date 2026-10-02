"""Runner pipeline eksperimen Rouen lengkap."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# path src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rouen_motion.config import load_config
from rouen_motion.pipeline import RouenMotionPipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete Rouen motion experiment pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.frames is not None:
        config.dataset.end_frame = args.frames
    pipeline = RouenMotionPipeline(config)

    print("=" * 70)

    print("ROUEN COMPUTER VISION EXPERIMENT PIPELINE")
    print(f"Dataset path:    {pipeline.dataset.frame_dir}")
    print(f"Total available: {len(pipeline.dataset.all_frame_paths)} frames")
    span_end = pipeline.config.dataset.end_frame if pipeline.config.dataset.end_frame is not None else len(pipeline.dataset)
    print(f"Configured span: {pipeline.config.dataset.start_frame} to {span_end}")
    print(f"Resolution:      {config.preprocessing.resize_width}x{config.preprocessing.resize_height}")

    print(f"Segmentation:    {config.segmentation.algorithm.upper()}")
    print(f"Tracking:        {config.tracking.method.upper()}")
    print(f"Optical Flow:    {config.optical_flow.method.upper()}")
    print("=" * 70)

    results = pipeline.run_all(max_frames=args.frames)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"[1] Segmentation:")
    print(f"    Regions detected:     {results['segmentation']['total_regions_detected']}")
    print(f"    Avg regions/frame:    {results['segmentation']['mean_regions_per_frame']:.1f}")
    print(f"    Speed:                {results['segmentation']['fps']} fps")

    print(f"\n[2] Motion Tracking:")
    print(f"    Unique tracks:        {results['tracking']['total_unique_tracks']}")
    print(f"    Avg active/frame:     {results['tracking']['mean_active_tracks_per_frame']:.1f}")
    print(f"    Occlusion recoveries: {results['tracking']['occlusion_recoveries']}")
    print(f"    Speed:                {results['tracking']['fps']} fps")

    print(f"\n[3] Optical Flow:")
    print(f"    Avg motion speed:     {results['optical_flow']['mean_flow_speed_px']:.2f} px/frame")
    print(f"    Max motion speed:     {results['optical_flow']['max_flow_speed_px']:.2f} px/frame")
    print(f"    Occlusion ratio:      {results['optical_flow']['mean_occlusion_pixel_ratio']*100:.1f}%")

    print(f"\n[4] Frame Interpolation (vs Ground Truth):")
    mc = results['interpolation']['motion_compensated']
    lb = results['interpolation']['linear_blend_baseline']
    gain = results['interpolation']['relative_gain']
    print(f"    Motion Comp PSNR:     {mc['mean_psnr_db']:.2f} dB (Linear: {lb['mean_psnr_db']:.2f} dB, Gain: {gain['psnr_improvement_db']:+.2f} dB)")
    print(f"    Motion Comp SSIM:     {mc['mean_ssim']:.4f} (Linear: {lb['mean_ssim']:.4f}, Gain: {gain['ssim_improvement']:+.4f})")
    print(f"    Motion Comp MAE:      {mc['mean_mae']:.2f} (Linear: {lb['mean_mae']:.2f})")

    print("\n" + "=" * 70)
    print(f"All visual figures and metrics saved to: {Path(config.output.directory).resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
