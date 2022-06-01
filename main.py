import open3d as o3d
import open3d.web_visualizer as o3d_web
import numpy as np
from dataset import Dataset
from reconstructor import DenseSLAMReconstructor, TSDFReconstructor, ReconstructorConfig

Reconstructor = DenseSLAMReconstructor

if __name__ == '__main__':
    dataset = Dataset.read("3d-scans-cap-black-glossy-ha-2019-02-27T16_06_29")
    device = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
    
    config = ReconstructorConfig(
        device=device, 
        camera_intrinsic=np.array([[613.688, 0.0, 323.035], 
                                [0.0, 614.261, 242.229], 
                                [0.0, 0.0, 1.0]]), 
        voxel_size=3.0 / 512.0, 
        block_resolution=16, 
        block_count=50000,
        depth_scale=1000.0,
        depth_max=3.0,
        depth_min=0.1,
        trunc_voxel_multiplier=8.0,
        odometry_distance_threshold=0.07,
    )

    reconstructor = Reconstructor(config)
    volume = reconstructor.reconstruct(dataset)
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_point_cloud("mesh.ply", mesh.to_legacy())
