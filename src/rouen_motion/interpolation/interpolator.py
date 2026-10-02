"""Interpolasi frame dengan motion compensation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from rouen_motion.config import InterpolationConfig
from rouen_motion.optical_flow.flow import BaseOpticalFlow, FarnebackOpticalFlow, warp_image_with_flow


@dataclass
class InterpolationResult:
    interpolated_frame: np.ndarray      # frame hasil interpolasi MC
    linear_blend_frame: np.ndarray      # baseline linear blend
    flow_forward: np.ndarray            # flow t -> t+1
    flow_backward: np.ndarray           # flow t+1 -> t
    alpha: float = 0.5
    metrics: Optional[Dict[str, float]] = None
    error_map: Optional[np.ndarray] = None


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Hitung PSNR (dB)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Hitung MAE."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Hitung SSIM pada kanal luminance."""
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    ksize = 11
    sigma = 1.5
    mu1 = cv2.GaussianBlur(g1, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(g2, (ksize, ksize), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 * g1, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (ksize, ksize), sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)

    return float(np.mean(ssim_map))


class MotionCompensatedInterpolator:
    """Interpolasi frame memakai optical flow dua arah."""

    def __init__(
        self,
        flow_estimator: Optional[BaseOpticalFlow] = None,
        config: Optional[InterpolationConfig] = None,
    ) -> None:
        self.flow_estimator = flow_estimator or FarnebackOpticalFlow()
        self.config = config or InterpolationConfig()

    def interpolate(
        self,
        frame0_bgr: np.ndarray,
        frame1_bgr: np.ndarray,
        alpha: float = 0.5,
        ground_truth: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> InterpolationResult:
        """Sintesis frame tengah pada posisi alpha."""
        gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)

        # forward flow: t -> t+1
        res_fwd = self.flow_estimator.compute(gray0, gray1)
        f_flow = res_fwd.flow

        # backward flow: t+1 -> t
        res_bwd = self.flow_estimator.compute(gray1, gray0)
        b_flow = res_bwd.flow

        # scale flow sesuai alpha
        flow_0_to_t = alpha * f_flow
        flow_1_to_t = (1.0 - alpha) * b_flow

        # warp frame
        warped0 = warp_image_with_flow(frame0_bgr, flow_0_to_t)
        warped1 = warp_image_with_flow(frame1_bgr, flow_1_to_t)

        # blending hasil warp
        interpolated = cv2.addWeighted(
            warped0,
            1.0 - alpha,
            warped1,
            alpha,
            0.0,
        )

        # baseline linear blend tanpa motion compensation
        linear_blend = cv2.addWeighted(
            frame0_bgr,
            1.0 - alpha,
            frame1_bgr,
            alpha,
            0.0,
        )

        metrics: Optional[Dict[str, float]] = None
        error_map: Optional[np.ndarray] = None

        if ground_truth is not None:
            # resize gt jika ukuran beda
            if ground_truth.shape[:2] != interpolated.shape[:2]:
                gt = cv2.resize(ground_truth, (interpolated.shape[1], interpolated.shape[0]))
            else:
                gt = ground_truth

            interp_psnr = compute_psnr(interpolated, gt)
            interp_ssim = compute_ssim(interpolated, gt)
            interp_mae = compute_mae(interpolated, gt)

            linear_psnr = compute_psnr(linear_blend, gt)
            linear_ssim = compute_ssim(linear_blend, gt)
            linear_mae = compute_mae(linear_blend, gt)

            error_map = np.abs(interpolated.astype(np.float32) - gt.astype(np.float32)).mean(axis=-1).astype(np.uint8)

            metrics = {
                "motion_psnr_db": interp_psnr,
                "motion_ssim": interp_ssim,
                "motion_mae": interp_mae,
                "linear_psnr_db": linear_psnr,
                "linear_ssim": linear_ssim,
                "linear_mae": linear_mae,
                "psnr_gain_db": interp_psnr - linear_psnr,
                "ssim_gain": interp_ssim - linear_ssim,
            }

            # evaluasi khusus ROI objek bergerak
            if roi_mask is not None and np.any(roi_mask):
                m_roi = roi_mask > 0
                mse_mc_roi = np.mean((interpolated[m_roi].astype(np.float64) - gt[m_roi].astype(np.float64)) ** 2)
                mse_lb_roi = np.mean((linear_blend[m_roi].astype(np.float64) - gt[m_roi].astype(np.float64)) ** 2)
                psnr_mc_roi = float(20.0 * np.log10(255.0 / np.sqrt(mse_mc_roi))) if mse_mc_roi > 0 else 100.0
                psnr_lb_roi = float(20.0 * np.log10(255.0 / np.sqrt(mse_lb_roi))) if mse_lb_roi > 0 else 100.0
                mae_mc_roi = float(np.mean(np.abs(interpolated[m_roi].astype(np.float64) - gt[m_roi].astype(np.float64))))
                mae_lb_roi = float(np.mean(np.abs(linear_blend[m_roi].astype(np.float64) - gt[m_roi].astype(np.float64))))

                metrics.update({
                    "roi_motion_psnr_db": psnr_mc_roi,
                    "roi_linear_psnr_db": psnr_lb_roi,
                    "roi_motion_mae": mae_mc_roi,
                    "roi_linear_mae": mae_lb_roi,
                    "roi_psnr_gain_db": psnr_mc_roi - psnr_lb_roi,
                })

        return InterpolationResult(
            interpolated_frame=interpolated,
            linear_blend_frame=linear_blend,
            flow_forward=f_flow,
            flow_backward=b_flow,
            alpha=alpha,
            metrics=metrics,
            error_map=error_map,
        )
