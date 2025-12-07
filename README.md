This repository contains my capstone work on 3D reconstruction from multiple 2D images. The project focuses on connecting the theory from *Multiple View Geometry* with practical experiments using real image data. The main components include fundamental matrix estimation, calibration matrix recovery, and homography computation. All scripts are self-contained and can be run directly.

## Structure and Usage

- **`collect_points.py`**  
  Opens two images and lets you manually pick corresponding points. Saves the selected points to a `.npy` file.

- **`visualize_points.py`**  
  Loads the saved `.npy` file and visualizes the correspondences across the stereo pair.

- **`estimate_F_and_K.py`**  
  Computes the fundamental matrix using the normalized 8-point algorithm and estimates the calibration matrix \(K\).

- **`compare_with_ground_truth_E.py`**  
  Uses the provided ground-truth \(K\) values to compute the essential matrix and compare it with the estimated version.

- **`refine_K_from_F.py`**  
  Performs nonlinear optimization to refine the calibration matrix using the structure of the essential matrix.

- **`homography_checkerboard.py`**  
  Detects checkerboard corners, computes homographies using both DLT and OpenCV, and visualizes the planar alignment.

Place your images in the `img_homo/` folder before running the scripts. Each script prints numerical results to the console and saves output files when needed.

This work forms the basis for my thesis, where I plan to continue exploring stereo reconstruction, planar geometry, and camera pose recovery.
