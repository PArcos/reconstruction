import open3d as o3d
import numpy as np
from reconstruction.dataset import Dataset
from reconstruction.config import Config
from reconstruction.reconstructor import Reconstructor 
import typer 


def main(data_dir: str = typer.Argument("data/3d-scans-cap-black-glossy-ha-2019-02-27T16_06_29"), 
    device: str = typer.Option(None),
    voxel_size: float = typer.Option(0.001),
    block_resolution: int = typer.Option(16),
    block_count: int = typer.Option(10000), 
    depth_scale: float = typer.Option(1000.0),
    depth_max: float = typer.Option(0.5),
    depth_min: float = typer.Option(0.01),
    fragment_size: int = typer.Option(8),
    max_correspondence_distance: float = typer.Option(0.015),
    icp_iterations: int = typer.Option(100),
    register_point_clouds: bool = typer.Option(False),
    match_images: bool = typer.Option(False)):
    """
    Performs reconstruction of the provided directory and visualizes results
    """
    dataset = Dataset.read(data_dir)

    # Set defaults
    if device is None:
        device = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"

    config = Config(
        device, 
        np.array([[613.688, 0.0, 323.035], 
                [0.0, 614.261, 242.229], 
                [0.0, 0.0, 1.0]]), 
        voxel_size, 
        block_resolution, 
        block_count,
        depth_scale,
        depth_max,
        depth_min,
        fragment_size,
        max_correspondence_distance,
        icp_iterations,
        register_point_clouds,
        match_images
    )

    config.device = device or config.device

    reconstructor = Reconstructor(config)
    volume = reconstructor.reconstruct(dataset)
    mesh = volume.extract_point_cloud(0.01)

    o3d.visualization.draw_geometries([mesh.to_legacy()])


if __name__ == "__main__":
    typer.run(main)
