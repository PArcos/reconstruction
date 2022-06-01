from typing import List, Union
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
from copy import deepcopy


def visualize_trajectory(transforms: List[np.ndarray]):

    original_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    original_mesh.scale(0.02, (0, 0, 0))

    meshes = []
    for t in transforms:
        mesh = deepcopy(original_mesh)
        mesh.transform(t)  
        meshes.append(mesh)

    visualize_geometry(meshes)


def visualize_geometry(geometry: List[o3d.geometry.PointCloud], flip: bool = False):
    if flip:
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        for g in geometry:
            g.transform(flip_transform)

    # TODO: We can use an offscreen renderer so we don't need to open a window
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for g in geometry:
        visualizer.add_geometry(g)
    visualizer.poll_events()
    visualizer.update_renderer()
    img = np.asarray(visualizer.capture_screen_float_buffer(False))
    
    figure = plt.figure(figsize=(10, 10))
    plt.axis(False)
    plt.imshow(img)
    plt.show()
    visualizer.destroy_window()