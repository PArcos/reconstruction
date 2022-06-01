

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
    """
    The number of fragments to use for each block."""
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True