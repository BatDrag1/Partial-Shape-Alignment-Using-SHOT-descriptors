import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from .rigid_transform import RigidTransform


def solver_point_to_point(
    scan: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]
) -> RigidTransform:
    """
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    """

    data_barycenter = scan.mean(axis=0)
    ref_barycenter = ref.mean(axis=0)
    covariance_matrix = (scan - data_barycenter).T.dot(ref - ref_barycenter)
    u, sigma, v = np.linalg.svd(covariance_matrix)
    rotation = v.T @ u.T

    # ensuring that we have a direct rotation (determinant equal to 1 and not -1)
    if np.linalg.det(rotation) < 0:
        u_transpose = u.T
        u_transpose[-1] *= -1
        rotation = v.T @ u_transpose

    translation = ref_barycenter - rotation.dot(data_barycenter)

    return RigidTransform(rotation, translation)


def solver_point_to_plane(
    scan: npt.NDArray[np.float64],
    ref: npt.NDArray[np.float64],
    normals_ref: npt.NDArray[np.float64],
) -> RigidTransform:
    g = np.hstack((np.cross(scan, normals_ref), normals_ref))
    h = np.einsum(
        "ij, ij->i",
        ref - scan,
        normals_ref,
    )
    # np.linalg.solve relies on a Cholesky decomposition when A is a symmetric definite positive matrix
    solution = np.linalg.solve(g.T @ g, g.T @ h)
    return RigidTransform(
        Rotation.from_euler("xyz", solution[:3]).as_matrix(), solution[3:6]
    )


def compute_point_to_point_error(
    scan: npt.NDArray[np.float64],
    ref: npt.NDArray[np.float64],
    transformation: RigidTransform,
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Computes the RMS error between a reference point cloud and data that went through the rigid transformation described
    by the rotation and the translation.
    """
    transformed_data = transformation[scan]
    distances = KDTree(ref).query(transformed_data)[0].squeeze()
    return np.sqrt((distances**2).mean()), transformed_data
