"""
Configuration dataclasses and YAML loader for the Rouen experiment project.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DatasetConfig:
    frame_dir: str = "data/rouen/frames"
    start_frame: int = 0
    end_frame: Optional[int] = None
    step: int = 1



@dataclass
class PreprocessingConfig:
    resize_width: int = 640
    resize_height: int = 360
    apply_gaussian_blur: bool = True
    blur_kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    blur_sigma: float = 0.0


@dataclass
class MOGConfig:
    history: int = 200
    var_threshold: float = 25.0
    detect_shadows: bool = True
    shadow_value: int = 127
    learning_rate: float = -1.0
    morph_kernel_size: List[int] = field(default_factory=lambda: [5, 5])
    min_contour_area: float = 80.0
    max_contour_area: float = 50000.0


@dataclass
class KMeansConfig:
    n_clusters: int = 4
    max_iter: int = 20
    epsilon: float = 1.0


@dataclass
class SegmentationConfig:
    algorithm: str = "mog"  # "mog" or "kmeans"
    mog: MOGConfig = field(default_factory=MOGConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)


@dataclass
class TrackingConfig:
    method: str = "multiframe"  # "twoframe" or "multiframe"
    iou_threshold: float = 0.25
    max_lost_frames: int = 6
    min_hits: int = 2
    velocity_damping: float = 0.95


@dataclass
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass
class MultiFrameFlowConfig:
    temporal_window: int = 3
    forward_backward_threshold: float = 1.5
    temporal_smoothing: bool = True


@dataclass
class OpticalFlowConfig:
    method: str = "farneback"
    farneback: FarnebackConfig = field(default_factory=FarnebackConfig)
    multiframe: MultiFrameFlowConfig = field(default_factory=MultiFrameFlowConfig)


@dataclass
class InterpolationConfig:
    enabled: bool = True
    alpha: float = 0.5
    evaluate_ground_truth: bool = True
    eval_skip_interval: int = 2


@dataclass
class VisualizationConfig:
    save_visualizations: bool = True
    show_vectors: bool = True
    vector_step: int = 20
    output_format: str = "png"


@dataclass
class OutputConfig:
    directory: str = "outputs"


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _populate_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Helper to populate dataclass recursively from nested dicts."""
    if not isinstance(data, dict):
        return data
    init_kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name in data:
            val = data[f.name]
            field_type = f.type
            if hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
                init_kwargs[f.name] = _populate_dataclass(field_type, val)
            else:
                init_kwargs[f.name] = val
    return cls(**init_kwargs)



def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Loads configuration from a YAML file.
    If no path is provided, default settings are returned.
    """
    if config_path is None:
        return AppConfig()

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    config = AppConfig()

    if "dataset" in raw_dict:
        config.dataset = _populate_dataclass(DatasetConfig, raw_dict["dataset"])
    if "preprocessing" in raw_dict:
        config.preprocessing = _populate_dataclass(PreprocessingConfig, raw_dict["preprocessing"])
    if "segmentation" in raw_dict:
        seg_dict = raw_dict["segmentation"]
        mog_cfg = _populate_dataclass(MOGConfig, seg_dict.get("mog", {}))
        km_cfg = _populate_dataclass(KMeansConfig, seg_dict.get("kmeans", {}))
        config.segmentation = SegmentationConfig(
            algorithm=seg_dict.get("algorithm", "mog"),
            mog=mog_cfg,
            kmeans=km_cfg
        )
    if "tracking" in raw_dict:
        config.tracking = _populate_dataclass(TrackingConfig, raw_dict["tracking"])
    if "optical_flow" in raw_dict:
        of_dict = raw_dict["optical_flow"]
        fb_cfg = _populate_dataclass(FarnebackConfig, of_dict.get("farneback", {}))
        mf_cfg = _populate_dataclass(MultiFrameFlowConfig, of_dict.get("multiframe", {}))
        config.optical_flow = OpticalFlowConfig(
            method=of_dict.get("method", "farneback"),
            farneback=fb_cfg,
            multiframe=mf_cfg
        )
    if "interpolation" in raw_dict:
        config.interpolation = _populate_dataclass(InterpolationConfig, raw_dict["interpolation"])
    if "visualization" in raw_dict:
        config.visualization = _populate_dataclass(VisualizationConfig, raw_dict["visualization"])
    if "output" in raw_dict:
        config.output = _populate_dataclass(OutputConfig, raw_dict["output"])

    return config
