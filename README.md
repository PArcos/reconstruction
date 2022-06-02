# Remy Robotics - 3D Reconstruction
## Pablo Arcos

We are asked to reconstruct a 3D model from a series of color & depth images gathered with a Realsense D435 camera. This is a typical Structure from Motion (SfM) problem with color and depth images. Normally this doesn't include motion but in this case we have the complete trajectory.

![Problem data](/docs/problem-data.jpg) 
![Reconstructed data](/docs/reconstructed.png) 

## How to install?
To install dependencies run `pip install -r requirements.txt`.

To run the CLI execute `python cli.py`
Check out `python cli.py --help` for more options.

## Where to start?
You can either run the CLI with `python cli.py --voxel-size=0.001` to try out the reconstruction or checkout the notebooks
* [Dataset](/notebooks/dataset_notebook.ipynb): Visualization & exploration of the provided data.
* [Reconstruction](/notebooks/reconstruction_notebook.ipynb): Exploration of reconstruction results 
* [Point cloud registration](/notebooks/point_cloud_registration.ipynb): Exploration of 3D data registration
* [Image matching](/notebooks/image_matching.ipynb): Exploration of image matching for visual odometry

## High level proposal
- [x] Build overlapping fragments
- [x] Register point cloud pairs to find local transform
- [ ] **(Explored)** Match image pairs to find local transform
- [ ] **(To do)** Build pose graph of estimated transforms
- [ ] **(To do)** Optimize it minizimizing reprojection error
- [x] Fuse fragments into a TSDF
- [x] Post-process to fix normals and improve mesh quality

## Future ideas
I can think of various ways to improve the result and the process and things to test. In no particular order:

* Run **hyper-parameter optimization** on the reconstruction. We can use [Optuna](https://optuna.org/) to find the best combination of parameters minimizing the reprojection error.
* Build a **pose graph** out of the fragments and use it to optimize them globally.
* **Try deep learning approaches**. There has been a lot of research in this area lately deriving from the [NeRF](https://arxiv.org/abs/2003.08934) paper, such as [ADOP](https://github.com/darglein/ADOP). We could also use it for the different stages such as [SuperPoint](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/superpoint.html) for keypoint detection or [DeepGlobalRegistration](https://github.com/chrischoy/DeepGlobalRegistration) for point cloud registration.
* **Improve speed** by implementing a real-time SLAM algorithm. Of the research I have seen I like the most [VILENS](https://ori.ox.ac.uk/labs/drs/vilens-tightly-fused-multi-sensor-odometry/) which uses plane tracking to speed up the 3D registration step and tricks to reduce the size of the pose graph. 
* **Detect salient object** automatically to remove unwanted parts of the mesh, we can use this paper [U2Net](https://github.com/xuebinqin/U-2-Net) 

## Resources
* [Open3D](http://www.open3d.org/docs/release/)
* [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
* [Typer](https://typer.tiangolo.com/)
