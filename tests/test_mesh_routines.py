import pytest
from numpy.testing import assert_allclose

import numpy as np
import pyvista
import trimesh

from openep.mesh.mesh_routines import (
    _create_trimesh,
    get_free_boundaries,
    calculate_mesh_volume,
    calculate_field_area,
    calculate_vertex_distance,
    calculate_vertex_path,
    mean_field_per_region,
    low_field_area_per_region,
)
from openep._datasets.simple_meshes import (
    CUBE, SPHERE, BROKEN_SPHERE, TRIANGLES
)


@pytest.fixture(scope='module')
def cube():
    return pyvista.read(CUBE)


@pytest.fixture(scope='module')
def sphere():
    return pyvista.read(SPHERE)


@pytest.fixture(scope='module')
def sphere_data(sphere):

    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    triangles = sphere.points[faces]
    areas = sphere.compute_cell_sizes(
            length=False,
            area=True,
            volume=False,
        )['Area']

    point_data = np.arange(sphere.n_points)
    sphere.point_data.set_array(point_data, 'data')
    cell_data = sphere.point_data_to_cell_data().cell_data['data']

    cell_region = sphere.cell_data['cell_region']
    unique_regions, region_weights = np.unique(cell_region, return_counts=True)

    return {
        "faces": faces,
        "triangles": triangles,
        "areas": areas,
        "point_data": point_data,
        "cell_data": cell_data,
        "cell_region": cell_region,
        "unique_regions": unique_regions,
        "region_weights": region_weights,
    }


@pytest.fixture(scope='module')
def broken_sphere():
    return pyvista.read(BROKEN_SPHERE)


@pytest.fixture(scope='module')
def triangles():
    return pyvista.read(TRIANGLES)


@pytest.fixture(scope='module')
def free_boundaries(triangles):

    return get_free_boundaries(triangles)


def test_create_trimesh(cube):

    faces = cube.faces.reshape(-1, 4)[:, 1:]
    trimesh_cube = _create_trimesh(cube)

    assert isinstance(trimesh_cube, trimesh.Trimesh)
    assert_allclose(cube.points, trimesh_cube.vertices)
    assert_allclose(faces, trimesh_cube.faces)


def test_get_free_boundaries(triangles, free_boundaries):

    # The first point of the triangles mesh is the centre of a square, and so not on a free boundary
    assert 0 not in free_boundaries.original_lines
    assert not np.any(np.all(triangles.points[0] == free_boundaries.points, axis=1))

    # All other points are part of free boundaries
    for point in triangles.points[1:]:
        assert np.any(np.all(point == free_boundaries.points, axis=1))

    assert 2 == free_boundaries.n_boundaries
    assert_allclose([5, 4], free_boundaries.n_points_per_boundary)


def test_FreeBoundary_calculate_areas(free_boundaries):

    # To calculate areas, we need to construct the boundary meshes based on the boundary points
    assert free_boundaries._boundary_meshes is None
    areas = free_boundaries.calculate_areas()
    assert_allclose([1, 0.5], areas)

    # Upon requesting the areas a second time, the boundary meshes should already be present
    new_areas = free_boundaries.calculate_areas()
    assert free_boundaries._boundary_meshes is not None
    assert_allclose(areas, new_areas)


def test_FreeBoundary_calculate_lengths(free_boundaries):

    square_perimeter = 4
    triangle_perimeter = 2 + np.sqrt(2)

    areas = free_boundaries.calculate_lengths()
    assert_allclose([square_perimeter, triangle_perimeter], areas)


def test_calculate_mesh_volume(sphere):

    volume = calculate_mesh_volume(sphere, fill_holes=False)
    assert_allclose(sphere.volume, volume)


def test_calculate_mesh_volume_repair(broken_sphere, sphere):

    volume = calculate_mesh_volume(broken_sphere, fill_holes=True)
    assert_allclose(sphere.volume, volume, atol=0.002)


def test_calculate_field_area(sphere, sphere_data):

    xbelow0 = sphere_data['triangles'][..., 0].mean(axis=1) <= 0  # select every triangle with mean x below YZ plane
    calculated_area = sphere_data['areas'][xbelow0].sum()

    area = calculate_field_area(sphere, sphere.points[:, 1], threshold=0)

    assert_allclose(calculated_area, area)


def test_calculate_vertex_distance_euclidian(cube):

    test_dist = calculate_vertex_distance(
        mesh=cube,
        start_index=0,
        end_index=7,
        metric='euclidian'
    )  # far corners are at indices 0 and 7

    assert_allclose(test_dist, np.sqrt(3), atol=5)


def test_calculate_vertex_distance_euclidian_sphere(sphere):

    sphere_diameter = 2.0
    start_index = 18  # top of sphere
    end_index = 23  # bottom of sphere
    test_dist = calculate_vertex_distance(
        mesh=sphere,
        start_index=start_index,
        end_index=end_index,
        metric='euclidian'
    )

    assert_allclose(test_dist, sphere_diameter, atol=5)


def test_calculate_vertex_distance_geodesic_sphere(sphere):

    sphere_half_circumference = 3.138363827815294  # should be equal to pi*diameter/2=pi, but it's not a perfect sphere
    start_index = 18  # top of sphere
    end_index = 23  # bottom of sphere
    test_dist = calculate_vertex_distance(
        mesh=sphere,
        start_index=start_index,
        end_index=end_index,
        metric='geodesic'
    )

    assert_allclose(test_dist, sphere_half_circumference, atol=5)


def test_calculate_vertex_distance_disconnected_regions(triangles):

    # These indices belong to two distinct components
    start_index = 0
    end_index = 5

    test_dist = calculate_vertex_distance(
        mesh=triangles,
        start_index=start_index,
        end_index=end_index,
    )

    assert np.isnan(test_dist)


def test_calculate_vertex_distance_invalid_metric(sphere):

    invalid_metric = "Manhattan"
    start_index = 18  # top of sphere
    end_index = 23  # bottom of sphere

    match = "metric must be on of: geodesic, euclidian"
    with pytest.raises(ValueError, match=match):
        calculate_vertex_distance(
            mesh=sphere,
            start_index=start_index,
            end_index=end_index,
            metric=invalid_metric,
        )


def test_calculate_vertex_path(cube):

    path = calculate_vertex_path(cube, 0, 7)
    assert_allclose(path, [0, 1, 7])


def test_calculate_vertex_path_disconnected(triangles):

    # These indices belong to two distinct components
    start_index = 0
    end_index = 5

    path = calculate_vertex_path(triangles, start_index, end_index)
    assert 0 == path.size


def test_mean_field_per_region(sphere, sphere_data):
    
    mean_value_per_region = mean_field_per_region(
        mesh=sphere,
        field=sphere_data['point_data'],
        cell_region=sphere_data['cell_region'],
    )

    # Check the weighted mean of the per-region data is equal to the total mean voltage
    _, weights = np.unique(sphere_data['cell_region'], return_counts=True)
    weighted_average = np.average(mean_value_per_region, weights=weights)

    assert mean_value_per_region.size == sphere_data['unique_regions'].size
    assert_allclose(np.nanmean(sphere_data['cell_data']), weighted_average, atol=1e-4, rtol=1e-4)


def test_mean_field_per_region_cell_data(sphere, sphere_data):
    
    mean_value_per_region = mean_field_per_region(
        mesh=sphere,
        field=sphere_data['cell_data'],
        cell_region=sphere_data['cell_region'],
    )

    # Check the weighted mean of the per-region data is equal to the total mean voltage
    _, weights = np.unique(sphere_data['cell_region'], return_counts=True)
    weighted_average = np.average(mean_value_per_region, weights=weights)

    assert mean_value_per_region.size == sphere_data['unique_regions'].size
    assert_allclose(np.nanmean(sphere_data['cell_data']), weighted_average, atol=1e-4, rtol=1e-4)

def test_low_field_area_per_region(sphere, sphere_data):

    low_value_area_per_region = low_field_area_per_region(
        mesh=sphere,
        field=sphere_data['point_data'],
        cell_region=sphere_data['cell_region'],
        threshold=sphere.n_points // 2,
    )

    assert low_value_area_per_region.size == sphere_data['unique_regions'].size
    assert_allclose(sphere.field_data['low_value_area'].item(), np.sum(low_value_area_per_region))

def test_low_field_area_per_region_add_area(sphere, sphere_data):

    sphere.cell_data.set_array(sphere_data['areas'], 'Area')

    low_value_area_per_region = low_field_area_per_region(
        mesh=sphere,
        field=sphere_data['point_data'],
        cell_region=sphere_data['cell_region'],
        threshold=sphere.n_points // 2,
    )

    _ = sphere.cell_data.pop('Area')

    assert low_value_area_per_region.size == sphere_data['unique_regions'].size
    assert_allclose(sphere.field_data['low_value_area'].item(), np.sum(low_value_area_per_region))


def test_low_field_area_per_region_cell_data(sphere, sphere_data):

    low_value_area_per_region = low_field_area_per_region(
        mesh=sphere,
        field=sphere_data['cell_data'],
        cell_region=sphere_data['cell_region'],
        threshold=sphere.n_points // 2,
    )

    assert low_value_area_per_region.size == sphere_data['unique_regions'].size
    assert_allclose(sphere.field_data['low_value_area'].item(), np.sum(low_value_area_per_region))
