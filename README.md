# Remy Robotics - 3D Reconstruction
## Pablo Arcos

We are asked to reconstruct a 3D model from a series of images and depth resources gathered with a Realsense D435 camera. This is a typical Structure from Motion (SfM) problem with color and depth images. Normally this doesn't include motion but in this case we have the complete trajectory.

![Problem data](/docs/problem-data.jpg)

## How to install?
To install dependencies run 'pip install -r requirements.txt'

## Where to start?
To see the reconstruction results you can go to the reconstruction notebook. From there I have different tests to implement 3D point cloud registration and image registration. Additionally, I have tests to do 


## High level proposal
- [x] Build overlapping fragments
- [x] Register point cloud pairs to find local transform
- [ ] (Explored) Match image pairs to find local transform
- [ ] (To do) Build pose graph of estimated transforms 
- [x] Fuse fragments into a TSDF

## Future ideas
I can think of various ways to improve the process and things to test.

## Vocabulary
* **Fragment:** Small sequence of consecutive frames
* **Pose graph:** Graph where nodes are transforms and edges are constraints
* **TSDF Volume:** Voxel volume where each voxel stores the distance to the nearest surface

## Resources
* [Open3D](http://www.open3d.org/docs/release/)
* [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
