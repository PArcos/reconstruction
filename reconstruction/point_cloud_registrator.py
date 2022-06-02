from reconstruction.config import Config
import open3d as o3d
import numpy as np
import logging 

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class PointCloudRegistrator:

    def __init__(self, config: Config):
        self.config = config

    def register(self, source: o3d.core.Tensor, target: o3d.core.Tensor, initial_transform: np.ndarray) -> np.ndarray:
        """
        Register a point cloud to another point cloud.
        Returns:
            Estimated relative transformation matrix            
        """
        # It needs normals to calculate the correspondences
        target.estimate_normals()
        
        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0000001,
                                        relative_rmse=0.0000001,
                                        max_iteration=self.config.icp_iterations)

        try:
            reg = o3d.t.pipelines.registration.icp(source, target, 
                                    self.config.max_correspondence_distance,
                                    np.linalg.inv(initial_transform), 
                                    estimation, 
                                    criteria, 
                                    voxel_size=self.config.voxel_size * 8)
        except RuntimeError as e:
            logging.warning(e)
            return initial_transform

        return np.linalg.inv(reg.transformation.numpy())