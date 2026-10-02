"""Visualisasi hasil eksperimen (segmentasi, tracking, flow, interpolasi)."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rouen_motion.segmentation.base import DetectedRegion
from rouen_motion.tracking.tracker import Track, TrackState
from rouen_motion.optical_flow.flow import FlowResult, flow_to_hsv


# warna distinct per track ID
def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Warna BGR konsisten per track ID."""
    np.random.seed(track_id * 31 + 7)
    color = np.random.randint(60, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


class ExperimentVisualizer:
    """
    Handles plotting and saving of computer vision experimental visualizations.
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_segmentation(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        regions: List[DetectedRegion],
        frame_idx: int,
        save_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Visualizes segmentation results:
        Panel 1: Original frame with detected moving region bounding boxes
        Panel 2: Cleaned binary foreground mask
        Panel 3: Mask overlaid with contours and centroids
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Panel 1: Detections overlaid on original frame
        overlay_rgb = frame_rgb.copy()
        for r in regions:
            x, y, bw, bh = r.bbox
            cv2.rectangle(overlay_rgb, (x, y), (x + bw, y + bh), (255, 60, 60), 2)
            cx, cy = int(r.centroid[0]), int(r.centroid[1])
            cv2.circle(overlay_rgb, (cx, cy), 3, (60, 255, 60), -1)
            cv2.putText(
                overlay_rgb,
                f"#{r.region_id} ({int(r.area)}px)",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Panel 3: Color-coded mask overlay
        color_mask = np.zeros_like(frame_rgb)
        color_mask[mask > 0] = [0, 220, 220]
        blended = cv2.addWeighted(frame_rgb, 0.65, color_mask, 0.35, 0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(overlay_rgb)
        axes[0].set_title(f"Frame {frame_idx:04d} - Detected Regions ({len(regions)})", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"Foreground Mask (MoG2)", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(blended)
        axes[2].set_title(f"Segmentation Overlay", fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "segmentation" / f"seg_frame_{frame_idx:04d}.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def visualize_tracking(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Track],
        frame_idx: int,
        save_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Visualizes object tracking with persistent IDs, bounding boxes,
        and temporal trajectory histories.
        """
        vis_bgr = frame_bgr.copy()

        for t in tracks:
            color = get_track_color(t.track_id)
            x, y, w, h = t.bbox

            # Draw trajectory tail
            if len(t.trajectory) > 1:
                pts = np.array(t.trajectory[-25:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_bgr, [pts], isClosed=False, color=color, thickness=2)

            # Draw bounding box
            is_occluded = t.state == TrackState.OCCLUDED
            thickness = 1 if is_occluded else 2
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), color, thickness=thickness)

            # Draw label
            status_tag = " [OCCL]" if is_occluded else ""
            label = f"ID:{t.track_id}{status_tag}"
            cv2.putText(
                vis_bgr,
                label,
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(vis_rgb)
        ax.set_title(
            f"Rouen Traffic Tracking - Frame {frame_idx:04d} | Active Tracks: {len(tracks)}",
            fontsize=13,
        )
        ax.axis("off")

        if save_path is None:
            save_path = self.output_dir / "tracking" / f"track_frame_{frame_idx:04d}.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def visualize_optical_flow(
        self,
        frame_curr_bgr: np.ndarray,
        flow_result: FlowResult,
        frame_idx: int,
        step: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Visualizes dense optical flow:
        Panel 1: Original frame
        Panel 2: Motion direction and magnitude encoded in HSV color-wheel
        Panel 3: Motion vectors (quiver arrows) overlaid on frame
        """
        frame_rgb = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_curr_bgr.shape[:2]

        # 1. Flow HSV color encoding
        flow_hsv_bgr = flow_to_hsv(flow_result.flow)
        flow_hsv_rgb = cv2.cvtColor(flow_hsv_bgr, cv2.COLOR_BGR2RGB)

        # 2. Flow quiver / vector field
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx = flow_result.flow[y, x, 0]
        fy = flow_result.flow[y, x, 1]
        mags = np.sqrt(fx**2 + fy**2)

        # Filter out static noise for quiver clarity
        motion_threshold = 0.5
        mask_motion = mags > motion_threshold

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Frame {frame_idx:04d} (Reference)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(flow_hsv_rgb)
        axes[1].set_title("Dense Optical Flow (HSV Color Wheel)", fontsize=12)
        axes[1].axis("off")

        # Quiver over grayscale image
        gray = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2GRAY)
        axes[2].imshow(gray, cmap="gray")
        if np.any(mask_motion):
            axes[2].quiver(
                x[mask_motion],
                y[mask_motion],
                fx[mask_motion],
                -fy[mask_motion],  # Flip Y for image coordinate plotting
                color="red",
                angles="xy",
                scale_units="xy",
                scale=0.5,
                width=0.003,
                headwidth=3.5,
            )
        axes[2].set_title(f"Motion Vectors (Step={step}px)", fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "optical_flow" / f"flow_frame_{frame_idx:04d}.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def visualize_interpolation(
        self,
        frame0_bgr: np.ndarray,
        interp_bgr: np.ndarray,
        frame1_bgr: np.ndarray,
        frame_idx_0: int,
        frame_idx_1: int,
        ground_truth: Optional[np.ndarray] = None,
        error_map: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Visualizes frame interpolation:
        Compares Original Frame t, Synthesized Frame t+0.5, and Original Frame t+1.
        If ground truth is provided, shows error map and quantitative metrics (PSNR, SSIM).
        """
        f0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
        synth_rgb = cv2.cvtColor(interp_bgr, cv2.COLOR_BGR2RGB)
        f1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)

        num_cols = 4 if (ground_truth is not None and error_map is not None) else 3
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 4.5))

        axes[0].imshow(f0_rgb)
        axes[0].set_title(f"Original Frame t ({frame_idx_0:04d})", fontsize=11)
        axes[0].axis("off")

        synth_title = "Synthesized Frame t+0.5"
        if metrics:
            synth_title += f"\nPSNR: {metrics.get('motion_psnr_db', 0):.2f}dB | SSIM: {metrics.get('motion_ssim', 0):.3f}"
        axes[1].imshow(synth_rgb)
        axes[1].set_title(synth_title, fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(f1_rgb)
        axes[2].set_title(f"Original Frame t+1 ({frame_idx_1:04d})", fontsize=11)
        axes[2].axis("off")

        if num_cols == 4:
            im = axes[3].imshow(error_map, cmap="inferno")
            axes[3].set_title("Reconstruction Error (|Synthesized - GT|)", fontsize=11)
            axes[3].axis("off")
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "interpolation" / f"interp_{frame_idx_0:04d}_{frame_idx_1:04d}.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
