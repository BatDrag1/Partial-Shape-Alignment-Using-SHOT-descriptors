import argparse
import gc
import logging
import warnings
from dataclasses import asdict
from pathlib import Path
import numpy as np

import coloredlogs

from scripts.parse_args import parse_args
from shot import (
    RegistrationPipeline,
    check_transform,
    checkpoint,
    compute_normals,
    get_data,
    get_incorrect_matches,
    get_transform_from_conf_file,
    load_config_from_yaml,
    RigidTransform,
)

warnings.filterwarnings("ignore")

def main(args: argparse.Namespace | None = None) -> None:
    """
    Runs a feature-based registration between two point clouds.
    Outputs useful information in stdout and writes ply files with the registered point clouds.

    Args:
        args: Arguments parsed from command-line using argparse.
    """
    coloredlogs.install(
        level="INFO",
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        field_styles={
            "levelname": {"color": "black", "bright": True, "bold": True},
            "asctime": {"color": "magenta", "bright": True},
        },
        level_styles={
            "info": {"color": "cyan", "faint": True},
            "critical": {"color": "red", "bold": True},
            "error": {"color": "red", "bright": True},
            "warning": {"color": "yellow", "bright": True},
        },
    )

    args = args or parse_args()
    configuration = load_config_from_yaml(args.config, vars(args))

    global_timer = checkpoint()
    timer = checkpoint()
    scan, scan_normals = get_data(
        args.scan_file_path,
        k=args.normals_computation_k,
        normals_computation_callback=compute_normals,
    )
    ref, ref_normals = get_data(
        args.ref_file_path,
        k=args.normals_computation_k,
        normals_computation_callback=compute_normals,
    )

    exact_transformation = None
    try:
        exact_transformation = get_transform_from_conf_file(
            args.conf_file_path, args.scan_file_path, args.ref_file_path
        )
    except (FileNotFoundError, KeyError):
        logging.warning(
            f"Conf file not found or incorrect under {args.conf_file_path}, ignoring it."
        )

    run_check_transform = False  # describes the covering between the two point clouds
    if run_check_transform:
        check_transform(scan, ref, exact_transformation)

    timer("Time spent retrieving the data")

    pipeline = RegistrationPipeline(
        scan=scan, scan_normals=scan_normals, ref=ref, ref_normals=ref_normals
    )
    pipeline.select_keypoints(**asdict(configuration["keypoint_selection"]))
    timer("Time spent selecting the key points")

    pipeline.compute_descriptors(
        **asdict(configuration["descriptor"]),
        disable_progress_bars=args.disable_progress_bars,
    )
    timer(
        f"Time spent computing the descriptors on the reference point cloud ({ref.shape[0]} points)"
    )
    gc.collect()

    pipeline.find_descriptors_matches(**asdict(configuration["matching"]))
    timer("Time spent finding matches between the descriptors")

    if exact_transformation is not None:
        correct_matches = get_incorrect_matches(
            pipeline.scan[pipeline.scan_keypoints][pipeline.matches[0]],
            pipeline.ref[pipeline.ref_keypoints][pipeline.matches[1]],
            exact_transformation,
        )
        logging.info(
            f"{correct_matches.sum()} correct matches out of {pipeline.scan_descriptors.shape[0]} descriptors."
        )

    transformation_ransac, inliers_ratio = pipeline.run_ransac(
        **asdict(configuration["ransac"]),
        exact_transformation=exact_transformation,
        disable_progress_bar=args.disable_progress_bars,
        ransac_algoritm = "updated",
    )
    timer("Time spent on RANSAC")
    logging.info(
        f"\nRatio of inliers pre-ICP: {inliers_ratio * 100:.2f}%"
    )
    logging.info(
        "-- Transformation after RANSAC --"
    )
    print(transformation_ransac)
    gc.collect()
    transformation_icp1, distance_to_map1, has_icp_converged = pipeline.run_updated_icp(
        transformation_ransac,
    )
    logging.info(
        "-- Transformation after the first found of ICP --"
    )
    print(transformation_icp1)

    transformation_icp2, distance_to_map2, has_icp_converged = pipeline.run_icp(
        transformation_init=transformation_icp1,
        **asdict(configuration["icp"]),
        disable_progress_bar=args.disable_progress_bars,
    )
    logging.info(
        "-- Transformation after the second found of ICP --"
    )
    print(transformation_icp2)
    

    timer("Time spent on ICP")
    overlap_before, inliers_ratio_before_transformation = pipeline.compute_metrics_post_icp(
        RigidTransform(np.eye(3), np.array([0, 0, 0])), args.d_max
    )
    overlap_ransac, inliers_ratio_post_ransac = pipeline.compute_metrics_post_icp(
        transformation_ransac, args.d_max
    )
    overlap_first_icp, inliers_ratio_post_icp1 = pipeline.compute_metrics_post_icp(
        transformation_icp1, args.d_max
    )
    overlap_final, inliers_ratio_post_icp2 = pipeline.compute_metrics_post_icp(
        transformation_icp2, args.d_max
    )
    logging.info(
        f"\nProportion of inliers before transformation: {inliers_ratio_before_transformation * 100:.2f}%"
        f"\nPercentage of overlap before transformation: {overlap_before * 100:.2f}%"
    )
    logging.info(
        f"\nProportion of inliers after RANSAC: {inliers_ratio_post_ransac * 100:.2f}%"
        f"\nPercentage of overlap after RANSAC: {overlap_ransac * 100:.2f}%"
    )
    logging.info(
        f"\nProportion of inliers after first ICP: {inliers_ratio_post_icp1 * 100:.2f}%"
        f"\nPercentage of overlap after first ICP: {overlap_first_icp * 100:.2f}%"
    )
    logging.info(
        f"\nRMS error on the ICP:  {distance_to_map2:.4f} (has converged: {'OK'[::has_icp_converged * 2 - 1]})."
        f"\nProportion of inliers: {inliers_ratio_post_icp2 * 100:.2f}%"
        f"\nPercentage of overlap: {overlap_final * 100:.2f}%"
    )


    if not args.disable_ply_writing:
        logging.info("")
        logging.info(
            " -- Writing the aligned points cloud in ply files under './data/results' --"
        )
        (results_folder := Path("../data/results")).mkdir(exist_ok=True, parents=True)
        file_name = (
            f"{Path(args.scan_file_path).stem}_on_{Path(args.ref_file_path).stem}"
        )
        pipeline.write_alignments(
            (
                str(results_folder / f"{file_name}_post_ransac.ply"),
                transformation_ransac,
            ),
            (str(results_folder / f"{file_name}_post_icp.ply"), transformation_icp2),
        )

    global_timer("\nTotal time spent")


if __name__ == "__main__":
    main()
