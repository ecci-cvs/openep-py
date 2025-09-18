# OpenEP
# Copyright (c) 2021 OpenEP Collaborators
#
# This file is part of OpenEP.
#
# OpenEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>

"""Module containing conduction velocity (CV) calculation methods and CV divergence method"""

import math
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from sklearn.neighbors import KDTree

from ..case.case_routines import interpolate_general_cloud_points_onto_surface


__all__ = ['preprocess_lat_egm', 'plane_fitting', 'divergence',
           'triangulation', 'radial_basis_function', 'exclude_collision_points']


def preprocess_lat_egm(
        case,
        include=None,
        lat_threshold=-10000
):
    """
    Preprocess case data for conduction velocity calculation by removing local activation times
    and corresponding bipolar electrogram points based on threshold and include.

    Args:
        case (Case): The case data containing electrogram and annotation information.

        include (np.ndarray, optional): A boolean array specifying which data points to include
                                        in the preprocessing for bipolar EGM data and LAT.
                                        Defaults to "includes" specified in the imported data.

        lat_threshold (int, optional): Threshold for local activation times. Points with local
                                       activation times less than this value will be removed.
                                       Defaults to -1000.
                                       If None, no threshold is applied.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - local_activation_time (np.ndarray): Array of cleaned local activation times.
               - bipolar_egm_pts (np.ndarray): Array of bipolar electrogram points corresponding
                 to the cleaned local activation times.

    """
    bipolar_egm_pts = case.electric.bipolar_egm.points[include]
    local_activation_time = (case.electric.annotations.local_activation_time[include]
                             - case.electric.annotations.reference_activation_time[include])

    if lat_threshold:
        bipolar_egm_pts = np.delete(bipolar_egm_pts, np.where(local_activation_time == lat_threshold), 0)
        local_activation_time = np.delete(local_activation_time, np.where(local_activation_time == lat_threshold))

    return local_activation_time, bipolar_egm_pts


def plane_fitting(
        bipolar_egm_pts,
        local_activation_time,
        case,
        leaf_size=5,
        min_n_nearest_neighbours=5,
        tree_query_radius=10
):
    """
    Fit a plane to bipolar electrogram points and calculate conduction velocity values and centroids.

    Args:
        bipolar_egm_pts (array):
            Array of bipolar electrogram points.

        local_activation_time (array):
            Array of local activation times.

        leaf_size (int, optional):
            Leaf size for KDTree. Defaults to 5.

        min_n_nearest_neighbours (int, optional):
            Minimum number of nearest neighbours. Defaults to 5.

        tree_query_radius (int, optional):
            Radius for tree query. Defaults to 10.

    Returns:
        tuple: Array of conduction velocities and array of triangle center points.

    Example 1:
        import openep

        case = openep.load_openep_mat("path/to/openep.mat")
        mesh = case.create_mesh()
        case.analyse.conduction_velocity.calculate(method='plane_fitting')

    Example 2:
        >> import numpy as np
        >> bipolar_egm_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], ...])
        >> local_activation_times = np.array([0.1, 0.2, 0.15, ...])
        >> cv_values, cv_centroids = plane_fitting(bipolar_egm_pts, local_activation_times)

        """
    cv_values = np.full(bipolar_egm_pts.shape[0], np.nan)
    cv_centroids = np.empty((bipolar_egm_pts.shape))
    ones_arr = np.ones(10)

    tree = KDTree(bipolar_egm_pts, leaf_size=leaf_size)
    all_nn_indices = tree.query_radius(bipolar_egm_pts, r=tree_query_radius)
    all_nns = [[bipolar_egm_pts[idx] for idx in nn_indices] for nn_indices in all_nn_indices]

    for egm_i in range(bipolar_egm_pts.shape[0]):

        if len(all_nns[egm_i]) >= min_n_nearest_neighbours:
            x1, x2, x3 = np.transpose(np.vstack((bipolar_egm_pts[egm_i], all_nns[egm_i])))
            m = np.append(local_activation_time[egm_i], local_activation_time[all_nn_indices[egm_i]])

            res = minimize(_obj_func, ones_arr, args=(x1, x2, x3, m))
            a, b, c, d, e, f, g, h, q, l = res.x

            x, y, z = bipolar_egm_pts[egm_i]

            tx = 2 * a * x + d * y + e * z + g
            ty = 2 * b * y + d * x + f * z + h
            tz = 2 * c * z + e * x + f * y + q

            vx = tx / (tx ** 2 + ty ** 2 + tz ** 2)
            vy = ty / (tx ** 2 + ty ** 2 + tz ** 2)
            vz = tz / (tx ** 2 + ty ** 2 + tz ** 2)

            cv_values[egm_i] = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            cv_centroids[egm_i] = [x, y, z]
        else:
            cv_centroids[egm_i] = bipolar_egm_pts[egm_i]

    return cv_values, cv_centroids


def triangulation(
    bipolar_egm_pts,
    local_activation_time,
    case,
    min_theta=30,
    electrode_distance_range=(1.5, 10),
    min_lat_difference=2,
    del_alpha=5
):
    """
    Perform triangulation on electrogram data to calculate conduction velocities.

    Args:
        bipolar_egm_pts (array):
            Array of bipolar electrogram points of size Nx3

        local_activation_time (array):
            Local activation times corresponding to bipolar_egm_pts.

        min_theta (float):
            Minimum angle threshold for triangle cell acceptance (degrees).

        electrode_distance_range (tuple):
            Range of acceptable electrode distances.

        min_lat_difference (float):
            Minimum local activation time difference.

        del_alpha (float):
            Alpha value for Delaunay triangulation.

    Returns:
        tuple: Array of conduction velocities and array of triangle center points.

    Example:
        import openep

        case = openep.load_openep_mat("path/to/openep.mat")
        mesh = case.create_mesh()
        case.analyse.conduction_velocity.calculate(method='triangulation')

    Reference:
        This method is based on the paper:
            Age-Related Changes in Human Left and Right Atrial Conduction by PIPIN KOJODJOJO M.R.C.P.
    """

    bi_egm_point_cloud = pv.PolyData(bipolar_egm_pts)
    bi_egm_surface = bi_egm_point_cloud.delaunay_3d(alpha=del_alpha)

    # Only extract triangle cell types from mesh, see pyvista.CellType.TRIANGLE (indexed at 5 in cells_dict)
    bi_egm_cells = bi_egm_surface.cells_dict[5]

    cv_values = np.full(bi_egm_cells.shape[0], np.nan)
    cv_centroids = np.empty(bi_egm_cells.shape)

    for cell_i in range(bi_egm_cells.shape[0]):

        # Find vtx id of cell and sort by LAT value
        vtx_id = bi_egm_cells[cell_i].astype(int)
        lat = local_activation_time[vtx_id]
        lat_sorted_id = np.argsort(lat)

        # Assign point O, A and B to the cell vtx
        lat_O, lat_A, lat_B = lat[lat_sorted_id]
        point_O, point_A, point_B = bipolar_egm_pts[vtx_id[lat_sorted_id]]

        # Calculate distance between OA, OB and AB
        dist_OA = np.linalg.norm(np.subtract(point_O, point_A))
        dist_OB = np.linalg.norm(np.subtract(point_O, point_B))
        dist_AB = np.linalg.norm(np.subtract(point_A, point_B))

        # Calculate difference in LAT between OA and OB
        tOA = lat_A - lat_O
        tOB = lat_B - lat_O

        # Calculate angles subtended by OA and OB
        theta = np.arccos((dist_OA**2 + dist_OB**2 - dist_AB**2) / (2 * dist_OA * dist_OB))

        # Conditions to check if triangle is acceptable
        if (math.degrees(theta) >= min_theta
                and electrode_distance_range[0] <= dist_OA <= electrode_distance_range[1]
                and electrode_distance_range[0] <= dist_OB <= electrode_distance_range[1]
                and tOA >= min_lat_difference
                and tOB >= min_lat_difference
        ):
            # Calculate angle subtended by OA and the direction of wavefront propagation
            alpha = np.arctan((tOB * dist_OA - tOA * dist_OB * np.cos(theta)) / (tOA * dist_OB * np.sin(theta)))
            cv_temp = (dist_OA / tOA) * np.cos(alpha)
            cv_values[cell_i] = cv_temp
        else:
            cv_values[cell_i] = np.nan

        centroid_i = np.mean([point_O, point_A, point_B], axis=0)
        cv_centroids[cell_i] = centroid_i

    # return np.array(cv_values), np.array(cv_centroids)
    return cv_values, cv_centroids


def _obj_func(coef, x1, x2, x3, m):
    """
    Objective function for plane fitting method.

    This function is used to fit a surface to the activations in the plane fitting method.
    It calculates the norm of the difference between the measured values 'm' and the
    values predicted by the quadratic surface defined by the coefficients.

    Parameters:
        coef (array): Coefficients of the quadratic surface (a, b, c, d, e, f, g, h, q, l).
        x1 (array): X-coordinates of the data points.
        x2 (array): Y-coordinates of the data points.
        x3 (array): Z-coordinates of the data points.
        m (array): Measured values at the data points.

    Returns:
        float: The norm of the difference between measured and predicted values.
    """
    a, b, c, d, e, f, g, h, q, l = coef
    predicted = (a * x1 ** 2 +
                 b * x2 ** 2 +
                 c * x3 ** 2 +
                 d * x1 * x2 +
                 e * x1 * x3 +
                 f * x2 * x3 +
                 g * x1 +
                 h * x2 +
                 q * x3 +
                 l)
    return np.linalg.norm(m - predicted)


def radial_basis_function(
        bipolar_egm_pts,
        local_activation_time,
        case,
        leaf_size=5,
        project_on_surface=True,
        epsilon=None,
        kernel='gaussian',
        smooth=1,
):
    """
        Calculate conduction velocities on a cardiac mesh using RBF interpolation.

    Args:
        bipolar_egm_pts (array):
            Array of bipolar electrogram points of size Nx3 representing the 3D coordinates.

        local_activation_time (array):
            Array of local activation times corresponding to the bipolar electrogram points.

        case (Case):
            Case to create Mesh object containing point coordinates and methods (e.g., for computing derivatives).

        project_on_surface (bool, optional):
            If True, projects the bipolar electrogram points onto the mesh surface before interpolation.
            Defaults to True.

        kernel (str, optional):
            Kernel type used in the RBF interpolation (e.g., 'gaussian').
            Defaults to 'gaussian'.

        epsilon (float, optional):
            Shape parameter for the RBF interpolation. Defaults to None.

        smooth (float, optional):
            Smoothing parameter for the RBF interpolation. Defaults to 1.

        leaf_size (int, optional):
            Leaf size parameter for the KDTree used in nearest neighbor queries.
            Defaults to 5.

    Returns:
        tuple:
            cv_values (ndarray):
              Array of conduction velocity values at the bipolar electrogram point locations.

            cv_centroids (ndarray):
              Array of point coordinates (Nx3) corresponding to the positions of the returned `cv_values`.
              Default these are the original `bipolar_egm_pts`.

            cv_interpolated (ndarray):
              Array of conduction velocity values interpolated at all mesh points.
    """
    mesh = case.create_mesh()

    if project_on_surface:
        tree = KDTree(mesh.points, leaf_size=2)
        dist, ind = tree.query(bipolar_egm_pts, k=1)
        points_of_interest = mesh.points[ind.flat]
    else:
        points_of_interest = bipolar_egm_pts

    rbf_interpolator = Rbf(
        points_of_interest[:, 0],
        points_of_interest[:, 1],
        points_of_interest[:, 2],
        local_activation_time,
        kernel=kernel,
        epsilon=epsilon,
        smooth=smooth
    )

    lat_interpolated = rbf_interpolator(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2])

    mesh['values'] = lat_interpolated
    deriv = mesh.compute_derivative('values')
    gradients = deriv['gradient']

    grad_norm_sq = gradients[:, 0] ** 2 + gradients[:, 1] ** 2 + gradients[:, 2] ** 2
    # CV at mesh points (already interpolated)
    cv_interpolated = 1 / np.sqrt(grad_norm_sq)

    tree = KDTree(mesh.points, leaf_size=leaf_size)
    _, indices = tree.query(bipolar_egm_pts, k=1)

    # We are using the cv_centroids at the bi_egm location
    # rather than the ones moved closer to the mesh
    # cv_centroids = mesh.points[indices.flatten()]
    cv_centroids = bipolar_egm_pts
    cv_values = cv_interpolated[indices.flatten()]

    return cv_values, cv_centroids, cv_interpolated


def divergence(
        case,
        bipolar_egm_pts,
        local_activation_time,
        collision_threshold=-1,
        focal_threshold=1,
        output_binary_field=False,
        interpolation_kws=None,
):
    """
    Calculate divergence of the conduction velocity (CV) field to identify regions of collision and
    focal activation within cardiac tissue based on local activation times (LATs) and electrogram points.

    Args:
        case (Case): The case object containing cardiac geometry and electrogram data.

        bipolar_egm_pts (np.ndarray): An array of coordinates for bipolar electrogram points.

        local_activation_time (np.ndarray): An array of local activation times corresponding to
                                            bipolar electrogram points.

        collision_threshold (float, optional): Divergence value below which regions are considered
                                               as collision fronts. Defaults to -1.

        focal_threshold (float, optional): Divergence value above which regions are considered as
                                           focal activation fronts. Defaults to 1.

        output_binary_field (bool, optional): If True, the output divergence field is binary, with 1
                                              indicating regions meeting the collision or focal threshold,
                                              and 0 otherwise. If False, the continuous divergence field
                                              is returned. Defaults to False.

        interpolation_kws (dict, optional): Keyword arguments for interpolation function.
                                            > interpolate_general_cloud_points_onto_surface(**interpolation_kws)
                                            Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
               - norm_cv_direction (np.ndarray): The normalized direction of conduction velocity
                 across the cardiac tissue surface.
               - divergence (np.ndarray): The divergence of the activation direction field. This
                 can either be a continuous field or a binary field indicating focal and collision
                 regions based on the 'output_binary_field' parameter.
    """

    temp_mesh = case.create_mesh()
    basic_mesh = pv.PolyData(temp_mesh.points, temp_mesh.faces)
    interpolation_kws = dict() if interpolation_kws is None else interpolation_kws

    tree = KDTree(temp_mesh.points, leaf_size=2)
    dist, ind = tree.query(bipolar_egm_pts, k=1)
    closest_mesh_points_to_egm_points = temp_mesh.points[ind.flat]

    interpolated_scalar = interpolate_general_cloud_points_onto_surface(
        case=case,
        cloud_values=local_activation_time,
        cloud_points=closest_mesh_points_to_egm_points,
        **interpolation_kws
    )

    basic_mesh['LAT_scalar'] = interpolated_scalar
    derivative = basic_mesh.compute_derivative(scalars='LAT_scalar')

    cv_direction = derivative['gradient'] / np.sum(derivative['gradient'] ** 2, axis=1)[:, np.newaxis]
    magnitude = np.sqrt(np.sum(cv_direction ** 2, axis=1))
    norm_cv_direction = cv_direction / magnitude[:, np.newaxis]

    basic_mesh['activation_direction'] = cv_direction
    div = basic_mesh.compute_derivative(scalars='activation_direction', divergence=True)
    divergence = div['divergence']

    if output_binary_field:
        divergence = np.where((divergence < collision_threshold) | (divergence > focal_threshold), 1, 0)

    return norm_cv_direction, divergence


def exclude_collision_points(
        mesh,
        cv_field,
        divergence_field,
        collision_threshold=-1,
        focal_threshold=1,
        radius=2,
):
    """
    Exclude conduction velocity values near regions of wave collision or focal discharge.

    Args:
        cv_field : np.ndarray
            Array of conduction velocity values corresponding to each mesh point.
        divergence_field : np.ndarray
            Array of divergence values at each mesh point.
        collision_threshold : float, optional
            Threshold for detecting wave collisions (default is -3).
        focal_threshold : float, optional
            Threshold for detecting focal discharges (default is 3).
        radius : float, optional
            Neighborhood radius within which to check for collision conditions (default is 4).

    Returns:

        np.ndarray
            Array of conduction velocity values, with values replaced by np.nan where a collision is detected.
    """
    cv_i_excluded = []
    tree = KDTree(mesh.points, leaf_size=2)
    neighbors_indices = tree.query_radius(mesh.points, r=radius)

    for i, neighbors in enumerate(neighbors_indices):
        # Check if any neighbor meets the collision criteria
        collision_found = any(
            divergence_field[j] >= focal_threshold or divergence_field[j] <= collision_threshold
            for j in neighbors
        )
        if collision_found:
            cv_i_excluded.append(np.nan)
        else:
            cv_i_excluded.append(cv_field[i])

    return np.array(cv_i_excluded)
