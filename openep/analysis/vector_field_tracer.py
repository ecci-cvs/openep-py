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

import pyvista as pv
import numpy as np

__all__ = ['VectorFieldTracer']

class VectorFieldTracer:
    """
    Calculate line/streamline visualisation of vector data for each cell data
    """
    @staticmethod
    def calculate_streamlines(
            mesh,
            direction,
            radius,
            **kwargs,
    ):
        """
        Uses V. Jacquemet's method to create evenly-spaced streamlines

        Arguments:
            mesh (pv.Polydata): Case object's mesh
            direction (np.ndarray) array of shape N_cells x 3 e.g. fibre vectors
            radius (float): the distance between streamlines will be larger than
                the radius and smaller than 2*radius

        Optional keyword-only args:
            seed_points (int): number of seed points tested to generate each
                streamline; the longest streamline is kept (default: 32)
            seed_region (int array): list of triangle indices among which seed
                points are picked (default: None, which means that all triangles
                are considered)
            orthogonal (bool): if True, rotate the orientation by 90 degrees
                (default: False)
            oriented_streamlines (bool): if True, streamlines only follow the
                vector field in the direction it points to (and not the opposite
                direction); the outputted streamlines are then oriented according
                to the vector field.
                If False (default), the orientation field is defined modulo pi
                instead of 2pi
            allow_tweaking_orientation (bool): if an orientation vector is parallel
                to an edge of the triangle, a small random perturbation is applied
                to that vector to satisfy the requirement (default: True);
                otherwise an exception is raised when the requirement is not
                satisfied
            singularity_mask_radius (float): when the orientation field has a
                singularity (e.g. focus or node), prevent streamlines from entering
                a sphere of radius 'singularity_mask_radius' x 'radius' around
                that singularity (default: 0.1)
            max_length (int): maximal number of iterations when tracing
                streamlines; it is needed because of nearly-periodic streamlines
                (default: 0, which means equal to the number of triangles)
            max_angle (float): stop streamline integration if the angle between
                two consecutive segments is larger than max_angle in degrees;
                0 means straight line and 180 means U-turn (default: 90)
            avoid_u_turns (bool): restrict high curvatures by maintaining a
                lateral (perpendicular) distance of at least 'radius' between
                a segment of the streamline and the other segments of the same
                streamline; this automatically sets 'max_angle' to at most
                90 degrees (default: True)
            random_seed (int): initialize the seed for pseudo-random number
                generation (default: seed based on clock)
            parallel (bool): if True (default), use multithreading wherever
                implemented
            num_threads (int): if possible, use that number of threads for parallel
                computing (default: let OpenMP choose)

            Returns:
        np.ndarray: Array of shape (N_lines*2, 3), containing start and end points of
        each fibre line segment, suitable for conversion into a PolyData with lines.

        Usage:
            The returned array can be passed to ``pyvista.Plotter.add_mesh`` to
            visualise fibre streamlines on the mesh.
        """
        from evenlyspacedstreamlines import evenly_spaced_streamlines

        # Obtain triangles from face
        faces = mesh.faces.reshape((-1, 4))
        assert np.all(faces[:, 0] == 3), "Not all faces on the mesh are triangles!"
        triangles = faces[:, 1:]

        list_of_lines, _, _ = evenly_spaced_streamlines(
            vertices=mesh.points,
            triangles=triangles,
            orientation=direction,
            radius=radius,
            **kwargs
        )

        # Create poly mesh using list_of_lines
        # This allows much faster load of lines (single add_mesh)
        cells = []
        offset = 0
        points = np.vstack(list_of_lines)
        for line in list_of_lines:
            n = len(line)
            cells.append(np.hstack(([n], np.arange(offset, offset + n))))
            offset += n
        poly = pv.PolyData(points)
        cells = np.hstack(cells).astype(np.int64)
        poly.lines = cells

        return poly

    @staticmethod
    def calculate_vector_lines(
            mesh,
            direction,
            mask_threshold
    ):
        """
        Adds fibre lines to each cell on the mesh.

        Arguments:
            mesh (pv.Polydata): Case object's mesh
            direction (np.ndarray) array of shape N_cells x 3 e.g. fibre vectors
            mask_threshold (int): Portion of cells to include the fibre lines on.

        Returns:
            np.ndarray: Array of shape (N_lines, 2, 3) or flattened (N_lines*2, 3),
            giving start and end points of each line segment.

        Usage:
            The returned array can be passed to ``pyvista.Plotter.add_lines`` to
            visualise fibre directions on the mesh.
        """
        cell_centers = mesh.cell_centers()

        # Not drawing lines for all cells (too many!)
        mask = np.random.rand(cell_centers.n_points) < mask_threshold

        # Calculate the start and end of the fibre line, and add mask.
        start_of_line = mesh.cell_centers().points[mask]
        end_of_line = np.add(mesh.cell_centers().points, direction)[mask]

        return np.hstack((start_of_line, end_of_line)).reshape(-1, 3)