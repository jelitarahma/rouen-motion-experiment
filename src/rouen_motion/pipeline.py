"""Pipeline orchestrator eksperimen motion Rouen."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from rouen_motion.config import AppConfig, load_config
from rouen_motion.dataset import RouenDataset
from rouen_motion.preprocessing import FramePreprocessor
from rouen_motion.segmentation import create_segmenter, BaseSegmenter
from rouen_motion.tracking import create_tracker, BaseTracker
from rouen_motion.optical_flow import create_optical_flow, BaseOpticalFlow
from rouen_motion.interpolation import MotionCompensatedInterpolator
from rouen_motion.visualization import ExperimentVisualizer


class RouenMotionPipeline:
    """Pipeline eksperimen video motion Rouen."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.dataset = RouenDataset(
            frame_dir=self.config.dataset.frame_dir,
            start_frame=self.config.dataset.start_frame,
            end_frame=self.config.dataset.end_frame,
            step=self.config.dataset.step,
        )
        self.preprocessor = FramePreprocessor(self.config.preprocessing)
        self.segmenter: BaseSegmenter = create_segmenter(self.config.segmentation)
        self.tracker: BaseTracker = create_tracker(self.config.tracking)
        self.optical_flow: BaseOpticalFlow = create_optical_flow(
            self.config.optical_flow,
            multi_frame=(self.config.tracking.method == "multiframe"),
        )
        self.interpolator = MotionCompensatedInterpolator(
            flow_estimator=self.optical_flow,
            config=self.config.interpolation,
        )
        self.visualizer = ExperimentVisualizer(self.config.output.directory)

    def run_segmentation_experiment(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Menjalankan eksperimen segmentasi pada urutan frame."""
        self.segmenter.reset()
        num_frames = min(len(self.dataset), max_frames or len(self.dataset))
        total_regions = 0
        per_frame_counts: List[int] = []
        t0 = time.time()

        for idx in range(num_frames):
            raw_frame, path = self.dataset[idx]
            smoothed_bgr, _ = self.preprocessor.process(raw_frame)
            seg_res = self.segmenter.segment(smoothed_bgr)
            count = len(seg_res.regions)
            per_frame_counts.append(count)
            total_regions += count

            # simpan visualisasi keyframe
            if self.config.visualization.save_visualizations and (idx < 5 or idx % 15 == 0):
                self.visualizer.visualize_segmentation(
                    frame_bgr=smoothed_bgr,
                    mask=seg_res.mask,
                    regions=seg_res.regions,
                    frame_idx=self.config.dataset.start_frame + idx * self.config.dataset.step,
                )

        elapsed = time.time() - t0
        metrics = {
            "experiment": "segmentation",
            "algorithm": self.config.segmentation.algorithm,
            "frames_processed": num_frames,
            "total_regions_detected": total_regions,
            "mean_regions_per_frame": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
            "max_regions_in_single_frame": int(np.max(per_frame_counts)) if per_frame_counts else 0,
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(num_frames / max(1e-4, elapsed), 2),
        }
        self._save_summary("segmentation_summary.json", metrics)
        return metrics

    def run_tracking_experiment(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Menjalankan eksperimen segmentasi + motion tracking antar frame."""
        self.segmenter.reset()
        self.tracker.reset()
        num_frames = min(len(self.dataset), max_frames or len(self.dataset))
        t0 = time.time()
        active_counts: List[int] = []
        seen_track_ids = set()

        for idx in range(num_frames):
            raw_frame, path = self.dataset[idx]
            smoothed_bgr, _ = self.preprocessor.process(raw_frame)
            seg_res = self.segmenter.segment(smoothed_bgr)
            tracks = self.tracker.update(seg_res.regions, frame_idx=idx)

            active_counts.append(len(tracks))
            for t in tracks:
                seen_track_ids.add(t.track_id)

            if self.config.visualization.save_visualizations and (idx < 5 or idx % 15 == 0):
                self.visualizer.visualize_tracking(
                    frame_bgr=smoothed_bgr,
                    tracks=tracks,
                    frame_idx=self.config.dataset.start_frame + idx * self.config.dataset.step,
                )

        elapsed = time.time() - t0
        occl_recoveries = getattr(self.tracker, "total_occlusion_recoveries", 0)

        metrics = {
            "experiment": "tracking",
            "tracker_type": self.config.tracking.method,
            "frames_processed": num_frames,
            "total_unique_tracks": len(seen_track_ids),
            "mean_active_tracks_per_frame": float(np.mean(active_counts)) if active_counts else 0.0,
            "occlusion_recoveries": occl_recoveries,
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(num_frames / max(1e-4, elapsed), 2),
        }
        self._save_summary("tracking_summary.json", metrics)
        return metrics

    def run_optical_flow_experiment(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Menjalankan estimasi optical flow dan visualisasi antar frame berurutan."""
        self.optical_flow.reset()
        num_frames = min(len(self.dataset), max_frames or len(self.dataset))
        if num_frames < 2:
            raise ValueError("Need at least 2 frames for optical flow")

        t0 = time.time()
        mean_magnitudes: List[float] = []
        max_magnitudes: List[float] = []
        occlusion_pixel_ratios: List[float] = []

        prev_raw, _ = self.dataset[0]
        _, prev_gray = self.preprocessor.process(prev_raw)

        for idx in range(1, num_frames):
            curr_raw, _ = self.dataset[idx]
            curr_bgr, curr_gray = self.preprocessor.process(curr_raw)

            flow_res = self.optical_flow.compute(prev_gray, curr_gray)
            mean_magnitudes.append(float(np.mean(flow_res.magnitude)))
            max_magnitudes.append(float(np.max(flow_res.magnitude)))

            if flow_res.occlusion_mask is not None:
                occl_ratio = float(np.mean(flow_res.occlusion_mask))
                occlusion_pixel_ratios.append(occl_ratio)

            if self.config.visualization.save_visualizations and (idx < 5 or idx % 15 == 0):
                self.visualizer.visualize_optical_flow(
                    frame_curr_bgr=curr_bgr,
                    flow_result=flow_res,
                    frame_idx=self.config.dataset.start_frame + idx * self.config.dataset.step,
                    step=self.config.visualization.vector_step,
                )

            prev_gray = curr_gray

        elapsed = time.time() - t0
        pairs_count = num_frames - 1

        metrics = {
            "experiment": "optical_flow",
            "method": self.config.optical_flow.method,
            "frame_pairs_processed": pairs_count,
            "mean_flow_speed_px": float(np.mean(mean_magnitudes)) if mean_magnitudes else 0.0,
            "max_flow_speed_px": float(np.max(max_magnitudes)) if max_magnitudes else 0.0,
            "mean_occlusion_pixel_ratio": float(np.mean(occlusion_pixel_ratios)) if occlusion_pixel_ratios else 0.0,
            "elapsed_seconds": round(elapsed, 3),
            "pairs_per_second": round(pairs_count / max(1e-4, elapsed), 2),
        }
        self._save_summary("optical_flow_summary.json", metrics)
        return metrics

    def run_interpolation_experiment(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Menjalankan eksperimen interpolasi frame:
        1. Sintesis frame perantara t+0.5 antara frame t dan t+1.
        2. Jika evaluasi ground-truth aktif, lewati frame t+1 dan lakukan sintesis
           dari frame t dan frame t+2, menghitung PSNR, SSIM, dan MAE secara kuantitatif
           terhadap frame t+1 aktual serta membandingkannya dengan baseline linear blending.
        """
        num_frames = min(len(self.dataset), max_frames or len(self.dataset))
        if num_frames < 3:
            raise ValueError("Need at least 3 frames for interpolation with ground truth evaluation")

        t0 = time.time()
        motion_psnr_list: List[float] = []
        linear_psnr_list: List[float] = []
        motion_ssim_list: List[float] = []
        linear_ssim_list: List[float] = []
        motion_mae_list: List[float] = []
        linear_mae_list: List[float] = []
        roi_motion_psnr_list: List[float] = []
        roi_linear_psnr_list: List[float] = []
        roi_motion_mae_list: List[float] = []
        roi_linear_mae_list: List[float] = []

        skip = self.config.interpolation.eval_skip_interval  # 2


        for idx in range(0, num_frames - skip, skip):
            f0_raw, _ = self.dataset[idx]
            f_mid_raw, _ = self.dataset[idx + 1]
            f2_raw, _ = self.dataset[idx + skip]

            f0, _ = self.preprocessor.process(f0_raw)
            f_mid_gt, _ = self.preprocessor.process(f_mid_raw)
            f2, _ = self.preprocessor.process(f2_raw)

            # mask ROI kendaraan
            seg_gt = self.segmenter.segment(f_mid_gt)
            roi_mask = seg_gt.mask if np.any(seg_gt.mask) else None

            # interpolasi frame t+1
            interp_res = self.interpolator.interpolate(
                frame0_bgr=f0,
                frame1_bgr=f2,
                alpha=self.config.interpolation.alpha,
                ground_truth=f_mid_gt,
                roi_mask=roi_mask,
            )

            if interp_res.metrics:
                motion_psnr_list.append(interp_res.metrics["motion_psnr_db"])
                linear_psnr_list.append(interp_res.metrics["linear_psnr_db"])
                motion_ssim_list.append(interp_res.metrics["motion_ssim"])
                linear_ssim_list.append(interp_res.metrics["linear_ssim"])
                motion_mae_list.append(interp_res.metrics["motion_mae"])
                linear_mae_list.append(interp_res.metrics["linear_mae"])

                if "roi_motion_psnr_db" in interp_res.metrics:
                    roi_motion_psnr_list.append(interp_res.metrics["roi_motion_psnr_db"])
                    roi_linear_psnr_list.append(interp_res.metrics["roi_linear_psnr_db"])
                    roi_motion_mae_list.append(interp_res.metrics["roi_motion_mae"])
                    roi_linear_mae_list.append(interp_res.metrics["roi_linear_mae"])

            if self.config.visualization.save_visualizations and (idx < 5 or idx % 10 == 0):
                self.visualizer.visualize_interpolation(
                    frame0_bgr=f0,
                    interp_bgr=interp_res.interpolated_frame,
                    frame1_bgr=f2,
                    frame_idx_0=self.config.dataset.start_frame + idx * self.config.dataset.step,
                    frame_idx_1=self.config.dataset.start_frame + (idx + skip) * self.config.dataset.step,
                    ground_truth=f_mid_gt,
                    error_map=interp_res.error_map,
                    metrics=interp_res.metrics,
                )

        elapsed = time.time() - t0
        eval_count = len(motion_psnr_list)

        metrics = {
            "experiment": "frame_interpolation",
            "evaluations_count": eval_count,
            "motion_compensated": {
                "mean_psnr_db": float(np.mean(motion_psnr_list)) if motion_psnr_list else 0.0,
                "mean_ssim": float(np.mean(motion_ssim_list)) if motion_ssim_list else 0.0,
                "mean_mae": float(np.mean(motion_mae_list)) if motion_mae_list else 0.0,
            },
            "linear_blend_baseline": {
                "mean_psnr_db": float(np.mean(linear_psnr_list)) if linear_psnr_list else 0.0,
                "mean_ssim": float(np.mean(linear_ssim_list)) if linear_ssim_list else 0.0,
                "mean_mae": float(np.mean(linear_mae_list)) if linear_mae_list else 0.0,
            },
            "relative_gain": {
                "psnr_improvement_db": float(np.mean(motion_psnr_list) - np.mean(linear_psnr_list)) if motion_psnr_list else 0.0,
                "ssim_improvement": float(np.mean(motion_ssim_list) - np.mean(linear_ssim_list)) if motion_ssim_list else 0.0,
            },
            "moving_vehicles_roi": {
                "motion_compensated": {
                    "mean_psnr_db": float(np.mean(roi_motion_psnr_list)) if roi_motion_psnr_list else 0.0,
                    "mean_mae": float(np.mean(roi_motion_mae_list)) if roi_motion_mae_list else 0.0,
                },
                "linear_blend_baseline": {
                    "mean_psnr_db": float(np.mean(roi_linear_psnr_list)) if roi_linear_psnr_list else 0.0,
                    "mean_mae": float(np.mean(roi_linear_mae_list)) if roi_linear_mae_list else 0.0,
                },
                "roi_psnr_difference_db": float(np.mean(roi_motion_psnr_list) - np.mean(roi_linear_psnr_list)) if roi_motion_psnr_list else 0.0,
            },
            "elapsed_seconds": round(elapsed, 3),
        }


        self._save_summary("interpolation_summary.json", metrics)
        return metrics

    def run_all(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Menjalankan seluruh pipeline eksperimen secara berurutan."""
        seg_res = self.run_segmentation_experiment(max_frames=max_frames)
        track_res = self.run_tracking_experiment(max_frames=max_frames)
        flow_res = self.run_optical_flow_experiment(max_frames=max_frames)
        interp_res = self.run_interpolation_experiment(max_frames=max_frames)

        full_results = {
            "dataset_info": {
                "path": str(self.dataset.frame_dir),
                "total_available_frames": len(self.dataset.all_frame_paths),
                "evaluated_frames": min(len(self.dataset), max_frames or len(self.dataset)),
            },
            "segmentation": seg_res,
            "tracking": track_res,
            "optical_flow": flow_res,
            "interpolation": interp_res,
        }
        self._save_summary("full_experiment_results.json", full_results)
        return full_results

    def _save_summary(self, filename: str, data: Dict[str, Any]) -> None:
        out_path = Path(self.config.output.directory) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
