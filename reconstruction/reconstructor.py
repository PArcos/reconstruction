from typing import Tuple, List
import open3d as o3d
from reconstruction.image_matcher import ImageMatcher
from reconstruction.point_cloud_registrator import PointCloudRegistrator
from reconstruction.dataset import Dataset
from reconstruction.config import Config
import numpy as np
import cv2 as cv
from itertools import accumulate
from more_itertools import windowed, pairwise
from reconstruction.utils.visualization import visualize_trajectory
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Reconstructor():
    def __init__(self, config: Config):
        self.config = config
        self.matcher = ImageMatcher(config)
        self.registrator = PointCloudRegistrator(config)

        self.device = o3d.core.Device(config.device)
        self.camera_intrinsic = o3d.core.Tensor(self.config.camera_intrinsic)
        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=config.voxel_size,
            block_resolution=config.block_resolution,
            block_count=config.block_count,
            device=self.device)

    def reconstruct(self, dataset: Dataset) -> o3d.t.geometry.VoxelBlockGrid:
        estimated_transforms = [f.get_transform() for f in dataset.frames]
        
        # 1. Create fragments
        # TODO: Not used currently
        log.debug(f"Building fragments of {self.config.fragment_size} frames")
        fragments = self._build_fragments(dataset)

        # 2. Match image keypoints
        # TODO: Implemented initial exploration

        # 3. Register point cloud pairs
        if self.config.register_point_clouds:
            estimated_transforms = self._register_point_clouds(dataset, estimated_transforms)

        # 4. Build pose graph from transformations and optimize it
        # TODO: Not implemented yet

        # 5. Fuse depth into TSDF volume
        logging.debug("Fusing depth into TSDF volume")
        self._fuse_depth(dataset, estimated_transforms)

        return self.volume


    def _build_fragments(self, dataset: Dataset) -> Tuple[list, list]:
        fragments = list(windowed(dataset.frames, self.config.fragment_size, step=self.config.fragment_size - 1))

        # Remove fill values from last fragment
        fragments[-1] = filter(lambda x: x is not None, fragments[-1])

        return fragments


    def _extract_point_cloud(self, depth_file: str, color_file: str) -> o3d.t.geometry.PointCloud:
        """
        Loads images and projects them in 3D.
        """
        depth_image = o3d.t.io.read_image(depth_file)
        color_image = o3d.t.io.read_image(color_file)
        rgbd_image = o3d.t.geometry.RGBDImage(color_image, depth_image)

        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.camera_intrinsic,
            depth_max=self.config.depth_max,
            depth_scale=self.config.depth_scale,
            with_normals=False)

        return pcd


    def _register_point_clouds(self, dataset: Dataset, estimated_transforms: List[np.ndarray]) -> List[np.ndarray]:
        """
        Registers point clouds using ICP.
        """
        # We use the initial transforms to refine them, so we can recreate the estimations
        estimated_transforms = estimated_transforms[:1]
        prev_point_cloud = None
        for source_frame, target_frame in pairwise(tqdm(dataset.frames)):
            log.debug('Registering point cloud {}/{}'.format(source_frame.id, len(dataset.frames)))

            # Optimization to avoid loading a point cloud twice
            if prev_point_cloud is None:
                source_point_cloud = self._extract_point_cloud(source_frame.depth_file, source_frame.color_file)
            else:
                source_point_cloud = prev_point_cloud
            target_point_cloud = self._extract_point_cloud(target_frame.depth_file, target_frame.color_file)
            target_point_cloud.estimate_normals()
            initial_transform = np.matmul(np.linalg.inv(source_frame.get_transform()), target_frame.get_transform())

            transform = self.registrator.register(source_point_cloud, target_point_cloud, initial_transform)
            estimated_transforms.append(np.matmul(source_frame.get_transform(), transform))
            prev_point_cloud = target_point_cloud

        return estimated_transforms


    def _fuse_depth(self, dataset: Dataset, estimated_transforms: List[np.ndarray]):
        """
        Fuses depth into TSDF volume.
        """
        for frame in tqdm(dataset.frames):
            color = o3d.t.io.read_image(frame.color_file).to(self.device)
            depth = o3d.t.io.read_image(frame.depth_file).to(self.device)

            transform = np.linalg.inv(estimated_transforms[frame.id])
            frustum_block_coords = self.volume.compute_unique_block_coordinates(
                depth, self.camera_intrinsic, transform, self.config.depth_scale,
                self.config.depth_max)

            self.volume.integrate(frustum_block_coords, depth, color,
                self.camera_intrinsic, self.camera_intrinsic, transform,
                self.config.depth_scale, self.config.depth_max)
