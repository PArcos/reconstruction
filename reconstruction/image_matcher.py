from typing import Tuple
from reconstruction.config import Config
import numpy as np
import cv2 as cv
import logging 

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ImageMatcher():
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = cv.SIFT_create()


    def match(self, source: np.ndarray, target: np.ndarray, initial_transform: np.ndarray) -> np.ndarray:
        """
        Match two images using RootSIFT.

        Returns:
            Estimated transformation.
        """
        # Based on https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        log.debug(f"Matching images {source.shape} and {target.shape}")

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

        log.debug(f"Found {len(source_keypoints)}/{len(good)} keypoints")
        source_keypoints = np.float32([source_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_keypoints = np.float32([target_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # We can only run 5-point algorithm if we have at least 5 matches
        if source_keypoints.shape[0] < 5:
            log.warning(f"Not enough matches found. {source_keypoints.shape[0]}")
            return initial_transform

        # We can build the essential matrix instead of the fundamental 
        # because we have the camera intrinsics
        essential, mask = cv.findEssentialMat(
            source_keypoints, target_keypoints, self.camera_intrinsic.numpy(), cv.RANSAC)
        rotation1, rotation2, translation = cv.decomposeEssentialMat(essential)

        # Build final transform
        # TODO The translation is a unit vector, it's just the direction of motion 
        translation *= np.linalg.norm(initial_transform[:3, 3])
        relative_transform = np.array(initial_transform, copy=True)
        relative_transform[:3, :3] = rotation1
        relative_transform[:3, 3:3] = translation

        return relative_transform


    def detect_keypoints(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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