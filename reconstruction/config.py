from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    device: str
    camera_intrinsic: np.ndarray
    voxel_size: float
    block_resolution: int
    block_count: int
    depth_scale: float
    depth_max: float
    depth_min: float
    fragment_size: int = 8
    max_correspondence_distance: float = 0.015
    icp_iterations: int = 100
    register_point_clouds: bool = False
    match_images: bool = False