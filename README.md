# Partial-Shape-Alignment-Using-SHOT-descriptors
Implementation of the SHOT algorithm introduced in "Unique Signatures of Histograms for Local Surface  Description" by Tomari et al. Used to create a program to align shapes only partially overlapping with each other. A reasonably overlap is needed for the implementation to work.

This repository is based on the implementation presented by aubin-tchoi (see https://github.com/aubin-tchoi/shot-fpfh/tree/main), but with focus on partial alignment. 
The main script can be found within `register_point_clouds.py`. The program consist of the following steps:
 - **Data retrieving**: Only ".ply" files of binary types are supported. The paths to the pointclouds can be placed within `parse_args.py` line 8 and 14. Line 8 contains the file to be transformed while line 14 contains the reference point cloud.
 - **Query point selection**: an algorithm taken from `query_points_selection.py` will be used to select a subset of query points.
 - **Descriptors computation**: The SHOT descriptors are computed for the selected keypoints.
 - **Descriptors matching**: Uses a algorithm in `matching.py` to match a subset of descriptors with one another.
 - **RANSAC**: An slightly alternative RANSAC in `ransac.py` is performed on the matches found. Performs a coarse registration.
 - **ICP**: ICP is performed in two different variations to get a finer registration.
 - **Data saving**: Two files each containing two pointclouds will be saved within `\data\results\...`. The first one is after the coarse alignment and the other after ICP.


## Basic usage
Install python poetry from: https://python-poetry.org/. Inside the terminal change the working directory to `Partial-Shape-Alignment-Using-SHOT-descriptors`. Write `poetry install` and `poetry run register_point_clouds` to run the program. Run `poetry run register_point_clouds --help` to get further information on arguments in the script.
This repository contains resources to compute SHOT, two 3D descriptors on point clouds. The main script register_point_clouds.py contains a pipeline consisting of the following steps:
