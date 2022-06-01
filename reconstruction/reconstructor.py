from typing import Tuple
from pydantic import BaseModel
import open3d as o3d
from reconstruction.dataset import Dataset
import numpy as np
import cv2 as cv
from itertools import accumulate
from more_itertools import windowed, pairwise
from reconstruction.utils.visualization import visualize_trajectory
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ReconstructorConfig(BaseModel):
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


class ClassicReconstructor():
    def __init__(self, config: ReconstructorConfig):
        self.config = config

        self.feature_extractor = cv.SIFT_create()
        self.device = o3d.core.Device(config.device)
        self.camera_intrinsic = o3d.core.Tensor(self.config.camera_intrinsic)
        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3d.core.float32, o3d.core.uint16, o3d.core.uint16),
            attr_channels=((1), (1), (3)),
            voxel_size=config.voxel_size,
            block_resolution=config.block_resolution,
            block_count=config.block_count,
            device=self.device)

    def reconstruct(self, dataset: Dataset) -> o3d.t.geometry.VoxelBlockGrid:
        # 1. Create fragments
        # fragments = list(windowed(dataset.frames, self.config.fragment_size, step=self.config.fragment_size - 1))

        # # Remove fill values from last fragment
        # fragments[-1] = filter(lambda x: x is not None, fragments[-1])

        # # 2. Extract image features
        # world_transforms = [np.identity(4)]

        # relative_transforms = []
        # for source_frame, target_frame in pairwise(dataset.frames):
        #     print('Registering image {}/{}'.format(source_frame.id, len(dataset.frames)))
        #     source_image = cv.imread(source_frame.color_file)
        #     target_image = cv.imread(target_frame.color_file)

        #     initial_transform = np.matmul(np.linalg.inv(source_frame.get_transform()), target_frame.get_transform())
        #     relative_transform = self._register_image_pair(source_image, target_image, initial_transform)
        #     relative_transforms.append(relative_transform)

        # world_transforms.extend(accumulate(relative_transforms, lambda world, rel: np.matmul(world, np.linalg.inv(rel))))

        # if self.config.debug:
        #     visualize_trajectory(world_transforms)

        # # 3. Register point cloud pairs
        # relative_transforms = [np.identity(4)]
        # world_transforms = []
        # for fragment in fragments:
        #     for source_frame, target_frame in pairwise(fragment):
        #         print('Registering point cloud {}/{}'.format(source_frame.id, len(dataset.frames)))
        #         source_point_cloud = self._extract_point_cloud(source_frame.depth_file, source_frame.color_file)
        #         target_point_cloud = self._extract_point_cloud(target_frame.depth_file, target_frame.color_file)

        #         initial_transform = np.matmul(np.linalg.inv(source_frame.get_transform()), target_frame.get_transform())
        #         relative_transform = self._register_pair(source_point_cloud, target_point_cloud, initial_transform)

        #         relative_transforms.append(relative_transform)

        #     world_transforms.extend(accumulate(relative_transforms, lambda world, rel: np.matmul(world, np.linalg.inv(rel))))

        # print(len(world_transforms))
        # if self.config.debug:
        #     visualize_trajectory(world_transforms)

        # 3. Integrate depth into TSDF volume
        for frame in tqdm(dataset.frames):
            color = o3d.t.io.read_image(frame.color_file).to(self.device)
            depth = o3d.t.io.read_image(frame.depth_file).to(self.device)

            transform = np.linalg.inv(frame.get_transform())
            frustum_block_coords = self.volume.compute_unique_block_coordinates(
                depth, self.camera_intrinsic, transform, self.config.depth_scale,
                self.config.depth_max)

            self.volume.integrate(frustum_block_coords, depth, color,
                                  self.camera_intrinsic, self.camera_intrinsic, transform,
                                  self.config.depth_scale, self.config.depth_max)

        return self.volume

    def _register_pair(self, source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud, initial_transform: np.ndarray) -> np.ndarray:
        """
        Register a point cloud to another using ICP.

        Returns:
            Estimated transformation.
        """
        estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
        criterias = [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                                                relative_rmse=0.0001,
                                                                max_iteration=50),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                0.00001, 0.00001, 30),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                0.000001, 0.000001, 14)
        ]
        max_correspondence_distances = o3d.utility.DoubleVector(
            [0.16, 0.08, 0.04])
        voxel_sizes = o3d.utility.DoubleVector([0.02, 0.01, 0.005])

        result = o3d.t.pipelines.registration.multi_scale_icp(source, target, voxel_sizes, criterias, max_correspondence_distances,
                                                              initial_transform, estimation)

        return result.transformation.numpy()

    def _match_image_pair(self, source: np.ndarray, target: np.ndarray, initial_transform: np.ndarray) -> np.ndarray:
        """
        Match two images using RootSIFT.

        Returns:
            Estimated transformation.
        """
        # Based on https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

        source_keypoints, source_descriptors = self._detect_keypoints(source)
        target_keypoints, target_descriptors = self._detect_keypoints(target)

        # Match keypoints
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)

        # Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32([source_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # We can only run 5-point algorithm if we have at least 5 matches
        if src_pts.shape[0] < 5:
            return initial_transform

        # We can build the essential matrix instead of the fundamental 
        # because we have the camera intrinsics
        essential, mask = cv.findEssentialMat(
            src_pts, dst_pts, self.camera_intrinsic.numpy(), cv.RANSAC)
        rotation1, rotation2, translation = cv.decomposeEssentialMat(essential)

        # Build final transform
        # TODO The translation is a unit vector, it's just the direction of motion 
        translation *= np.linalg.norm(initial_transform[:3, 3])
        relative_transform = np.array(initial_transform, copy=True)
        relative_transform[:3, :3] = rotation1
        relative_transform[:3, 3:3] = translation

        return relative_transform


    def _detect_keypoints(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects keypoints in an image.

        Args:
            image: RGB image.
        Returns:
            Keypoints and descriptors.
        """
        # Implements RootSIFT (https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
        eps = 1e-7

        (kps, descs) = self.feature_extractor.detectAndCompute(image, None)
        
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return (kps, descs)


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
