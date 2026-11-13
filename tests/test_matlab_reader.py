"""MATLAB metadata and unipolar component loading."""

from copy import deepcopy

import h5py
import numpy as np
import pytest
import scipy.io

import openep
from openep.data_structures.electric import extract_electric_data
from openep.io.matlab import _load_mat_v73


def write_labels(file, key, labels):
    references = np.empty(labels.shape, dtype=h5py.ref_dtype)
    for index in np.ndindex(labels.shape):
        label = labels[index]
        name = '#refs#/label-' + '-'.join(map(str, index))
        if label:
            values = np.frombuffer(label.encode('utf-16-le'), dtype='<u2')
            target = file.create_dataset(name, data=values.reshape(-1, 1))
        else:
            target = file.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
            target.attrs['MATLAB_empty'] = 1
        target.attrs['MATLAB_class'] = np.bytes_('char')
        references[index] = target.ref
    dataset = file.create_dataset(key, data=references)
    dataset.attrs['MATLAB_class'] = np.bytes_('cell')


@pytest.mark.parametrize('matlab_class', ['double', 'cell'])
@pytest.mark.parametrize('key', [
    'notes', 'electric/ecgNames', 'electric/names', 'electric/tags',
    'electric/electrodeNames_bip', 'electric/electrodeNames_uni',
])
def test_matlab_empty_string_metadata(tmp_path, key, matlab_class):
    filename = tmp_path / 'empty.mat'
    with h5py.File(filename, 'w') as file:
        dataset = file.create_dataset('userdata/' + key, data=np.array([0, 0], dtype=np.uint64))
        dataset.attrs['MATLAB_class'] = np.bytes_(matlab_class)
        dataset.attrs['MATLAB_empty'] = 1
    data = _load_mat_v73(filename)
    for part in key.split('/'):
        data = data[part]
    assert data.size == 0
    assert data.dtype.kind == 'U'


def test_empty_direct_string_metadata(tmp_path):
    filename = tmp_path / 'empty-string.mat'
    with h5py.File(filename, 'w') as file:
        dataset = file.create_dataset('userdata/cartoFolder', data=np.array([0, 0], dtype=np.uint64))
        dataset.attrs['MATLAB_class'] = np.bytes_('char')
        dataset.attrs['MATLAB_empty'] = 1
    assert _load_mat_v73(filename)['cartoFolder'] == ''


@pytest.mark.parametrize('n_points', [1, 3])
@pytest.mark.parametrize('n_components', [1, 2, 3])
def test_unipolar_label_dimensions(tmp_path, n_points, n_components):
    labels = np.array([
        [f'point-{point} component-{component}' for point in range(n_points)]
        for component in range(n_components)
    ])
    filename = tmp_path / 'labels.mat'
    with h5py.File(filename, 'w') as file:
        write_labels(file, 'userdata/electric/electrodeNames_uni', labels)
    actual = _load_mat_v73(filename)['electric']['electrodeNames_uni']
    np.testing.assert_array_equal(actual, labels.T)


def test_nonempty_and_blank_labels(tmp_path):
    filename = tmp_path / 'labels.mat'
    with h5py.File(filename, 'w') as file:
        write_labels(file, 'userdata/electric/ecgNames', np.array([['ECG III', '', 'ECG II']]))
    np.testing.assert_array_equal(_load_mat_v73(filename)['electric']['ecgNames'], ['ECG III', '', 'ECG II'])


def test_unmarked_numeric_names_are_not_treated_as_empty(tmp_path):
    filename = tmp_path / 'invalid.mat'
    with h5py.File(filename, 'w') as file:
        file.create_dataset('userdata/electric/ecgNames', data=np.array([[np.nan]]))
    with pytest.raises(TypeError):
        _load_mat_v73(filename)


@pytest.fixture
def electric_data():
    def create(n_components):
        n_points, n_samples = 3, 4
        waves = np.arange(n_points * n_samples * n_components, dtype=float).reshape(n_points, n_samples, n_components)
        coordinates = np.arange(n_points * 3 * n_components, dtype=float).reshape(n_points, 3, n_components)
        return {
            'egm': np.arange(n_points * n_samples, dtype=float).reshape(n_points, n_samples),
            'egmX': np.arange(n_points * 3, dtype=float).reshape(n_points, 3),
            'egmRef': np.ones((n_points, n_samples)),
            'egmUni': waves[:, :, 0] if n_components == 1 else waves,
            'egmUniX': coordinates[:, :, 0] if n_components == 1 else coordinates,
            'electrodeNames_bip': np.array(['A-B', 'C-D', 'E-F']),
            'electrodeNames_uni': np.array([
                [f'point-{point} component-{component}' for component in range(n_components)]
                for point in range(n_points)
            ], dtype=object),
            'voltages': {'bipolar': np.arange(n_points, dtype=float), 'unipolar': np.arange(n_points, dtype=float)},
            'include': np.array([1, 0, 1]),
            'tags': np.array(['', '', '']),
            'names': np.array(['point-0', 'point-1', 'point-2']),
            'ecg': np.array([]),
            'ecgNames': np.array([], dtype=str),
            'impedances': {'time': np.full(n_points, np.nan), 'value': np.full(n_points, np.nan)},
            'egmSurfX': np.zeros((n_points, 3)),
            'barDirection': np.full((n_points, 3), np.nan),
            'annotations': {'mapAnnot': np.array([1, 2, 3]), 'referenceAnnot': np.array([1, 1, 1]),
                            'woi': np.array([[0, 3], [0, 3], [0, 3]])},
            'sampleFrequency': 1000.0,
        }
    return create


@pytest.mark.parametrize('n_components', [1, 2, 3])
@pytest.mark.parametrize('gain_kind', ['missing', 'empty', 'per_point', 'per_point_column', 'per_component'])
def test_unipolar_component_alignment(electric_data, n_components, gain_kind):
    data = electric_data(n_components)
    if gain_kind in ('per_point', 'per_point_column'):
        data['egmUniGain'] = np.array([1., 2., 3.])
        gains = np.tile(data['egmUniGain'][:, None], (1, n_components))
        if gain_kind == 'per_point_column':
            data['egmUniGain'] = data['egmUniGain'][:, None]
    elif gain_kind == 'per_component':
        gains = np.arange(3 * n_components, dtype=float).reshape(3, n_components)
        data['egmUniGain'] = gains.copy()
    else:
        gains = np.zeros((3, n_components))
        if gain_kind == 'empty':
            data['egmUniGain'] = np.array([])
    original = deepcopy(data)
    electric = extract_electric_data(data)
    np.testing.assert_array_equal(electric.unipolar_egm.egm, original['egmUni'])
    np.testing.assert_array_equal(electric.unipolar_egm.points, original['egmUniX'])
    np.testing.assert_array_equal(electric.unipolar_egm.names, original['electrodeNames_uni'].astype(str))
    np.testing.assert_array_equal(electric.unipolar_egm.gain, gains)
    np.testing.assert_array_equal(electric.bipolar_egm.egm, original['egm'])
    np.testing.assert_array_equal(electric.reference_egm.egm, original['egmRef'])
    np.testing.assert_array_equal(electric.include, original['include'])
    np.testing.assert_array_equal(electric.annotations._local_activation_time_indices, original['annotations']['mapAnnot'])
    assert np.isnan(electric.impedance.times).all()
    assert np.isnan(electric.impedance.values).all()
    assert np.isnan(electric.surface.normals).all()
    assert electric.ecg.ecg is None
    assert electric.ecg.channel_names is None
    assert electric.ecg.n_channels == 0


@pytest.mark.parametrize('n_components', [1, 2, 3])
def test_blank_unipolar_cells(electric_data, n_components):
    data = electric_data(n_components)
    data['electrodeNames_uni'][1, -1] = np.array([], dtype=np.uint16)
    electric = extract_electric_data(data)
    assert electric.unipolar_egm.names.shape == (3, n_components)
    assert electric.unipolar_egm.names[1, -1] == ''
    assert electric.unipolar_egm.names[0, 0] == 'point-0 component-0'


@pytest.mark.parametrize('n_components', [1, 2, 3])
@pytest.mark.parametrize('missing', [True, False])
def test_unavailable_unipolar_names(electric_data, n_components, missing):
    data = electric_data(n_components)
    if missing:
        del data['electrodeNames_uni']
    else:
        data['electrodeNames_uni'] = np.array([])
    electric = extract_electric_data(data)
    np.testing.assert_array_equal(electric.unipolar_egm.names, np.full((3, n_components), '', dtype=str))


@pytest.mark.parametrize('field,value', [
    ('electrodeNames_uni', np.array([['A', 'B'], ['C', 'D'], ['E', 'F']])),
    ('electrodeNames_uni', np.array(['A', 'B', 'C'])),
    ('egmUniX', np.zeros((3, 3, 2))),
    ('egmUniGain', np.ones((3, 2))),
    ('egmUniGain', np.ones(2)),
])
def test_inconsistent_unipolar_metadata(electric_data, field, value):
    data = electric_data(3)
    data[field] = value
    with pytest.raises(ValueError, match=field):
        extract_electric_data(data)


def test_single_component_squeezed_names(electric_data):
    data = electric_data(1)
    data['electrodeNames_uni'] = data['electrodeNames_uni'].ravel()
    electric = extract_electric_data(data)
    assert electric.unipolar_egm.names.shape == (3, 1)


def test_missing_ecg_names_without_ecgs(electric_data):
    data = electric_data(2)
    del data['ecgNames']
    electric = extract_electric_data(data)
    assert electric.ecg.ecg is None
    assert electric.ecg.channel_names is None


def test_exported_ecg_order(electric_data):
    data = electric_data(2)
    data['ecg'] = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    data['ecgNames'] = np.array(['ECG III', 'ECG II'])
    original = data['ecg'].copy()
    electric = extract_electric_data(data)
    np.testing.assert_array_equal(electric.ecg.ecg, original)
    np.testing.assert_array_equal(electric.ecg.channel_names, ['ECG III', 'ECG II'])


@pytest.mark.parametrize('n_components', [1, 2, 3])
def test_electrical_point_mask_preserves_alignment(electric_data, n_components):
    data = electric_data(n_components)
    data['egm'][1, :] = np.nan
    original = deepcopy(data)
    electric = extract_electric_data(data)
    for actual, field in [(electric.unipolar_egm.egm, 'egmUni'),
                          (electric.unipolar_egm.points, 'egmUniX'),
                          (electric.unipolar_egm.names, 'electrodeNames_uni')]:
        np.testing.assert_array_equal(actual, original[field][[0, 2]])
    assert electric.unipolar_egm.gain.shape == (2, n_components)


@pytest.mark.parametrize('n_components', [1, 2, 3])
def test_v7_loading_and_export_round_trip(tmp_path, electric_data, n_components):
    data = electric_data(n_components)
    surface = {
        'triRep': {'X': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                   'Triangulation': np.array([[1., 3., 2.], [1., 2., 4.], [2., 3., 4.], [3., 1., 4.]])},
        'act_bip': np.arange(8, dtype=float).reshape(4, 2),
        'uni_imp_frc': np.arange(12, dtype=float).reshape(4, 3),
    }
    filename = tmp_path / 'synthetic-v7.mat'
    scipy.io.savemat(filename, {'userdata': {'surface': surface, 'electric': data, 'notes': 'Synthetic case'}})
    original = deepcopy(data)
    case = openep.load_openep_mat(filename)
    exported_filename = tmp_path / 'exported-v7.mat'
    openep.export_openep_mat(case, exported_filename)
    reloaded = openep.load_openep_mat(exported_filename)
    for actual in (case, reloaded):
        np.testing.assert_array_equal(actual.points, surface['triRep']['X'])
        np.testing.assert_array_equal(actual.indices, surface['triRep']['Triangulation'] - 1)
        np.testing.assert_array_equal(actual.electric.unipolar_egm.egm, original['egmUni'])
        np.testing.assert_array_equal(actual.electric.unipolar_egm.points, original['egmUniX'])
        np.testing.assert_array_equal(actual.electric.unipolar_egm.names, original['electrodeNames_uni'].astype(str))
        assert actual.electric.unipolar_egm.gain.shape == (3, n_components)
        assert actual.electric.ecg.ecg is None
        assert actual.electric.ecg.channel_names is None
