This repository contains my capstone work on 3D reconstruction from multiple 2D images. The project focuses on understanding the core ideas from Multiple View Geometry and implementing small experiments that connect the theory to actual image data. The main experiments include estimating a fundamental matrix from manually selected correspondences, computing a camera calibration matrix, and recovering planar homographies using real images. All scripts are self-contained and meant to be run directly.

Included in this repository:

Code for collecting point correspondences from stereo image pairs

Fundamental matrix estimation (normalized 8-point algorithm)

Calibration matrix estimation + refinement

Homography estimation using DLT and OpenCV

Visualizations for checking alignment and warp quality

A written report summarizing the theory and experiments

To run the experiments, place your images in the specified folders and execute the corresponding Python scripts. Each script prints results to the console and saves output files where needed.
