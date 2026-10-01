from attr import attrs
import numpy as np

__all__ = []

@attrs(auto_attribs=True, auto_detect=True)
class Vectors:
    """
    Class for storing information about arrows/vectors and lines on surface

    Args:
        fibres (np.ndarray): array of shape N_cells x 3
        divergence (np.ndarray): array of shape N_cells x 3
        linear_connections (np.ndarray): array of shape M x 3 (represents the linear connections between endo and epi)
        linear_connection_regions (np.ndarray): array of shape N_cells
    """

    # TODO: move divergence arrows into Arrows class
    # TODO: remove longitudinal and transversal arrows from Fields class
    _fibres: np.ndarray = None
    divergence: np.ndarray = None
    linear_connections: np.ndarray = None
    linear_connection_regions: np.ndarray = None
    CUSTOM_VECTORS = list()

    def __attrs_post_init__(self):
        self.CUSTOM_VECTORS.clear()

    def __repr__(self):
        return f"vectors: {tuple(self.__dict__.keys())}"

    def __getitem__(self, arrow):
        try:
            return self.__dict__[arrow]
        except KeyError:
            raise ValueError(f"There is no vector '{arrow}'.")

    def __setitem__(self, arrow, value):
        if arrow not in type(self).__annotations__ and arrow not in self.CUSTOM_VECTORS:
            self.CUSTOM_VECTORS.append(arrow)

        self.__dict__[arrow] = value

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __contains__(self, arrow):
        return arrow in self.__dict__.keys()

    @property
    def fibres(self):
        return self._fibres

    @fibres.setter
    def fibres(self, array):
        self._fibres = None if np.all(array == [1, 0, 0]) else array

    @property
    def linear_connection_regions_names(self):
        if self.linear_connection_regions is None:
            return []
        regions = np.unique(self.linear_connection_regions).astype(str)
        return regions.tolist()

    def copy(self):
        """Create a deep copy of Arrows"""

        arrows = Vectors()
        for arrow in self:
            if self[arrow] is None:
                continue
            arrows[arrow] = np.array(self[arrow])

        return arrows

    @property
    def custom(self):
        return {key: self.__dict__[key] for key in self.CUSTOM_VECTORS if key in self.__dict__}


def extract_vector_data(surface_data, indices):
    """Extract vector data from surface data dictionary.

    Args:
        surface_data (dict): Dictionary containing numpy arrays that describe the
            surface of a mesh as well as scalar values (fields)
        indices (ndarray): Indices of points that make up each face of the mesh

    Returns:
        vectors (Vectors): Class for storing information about arrows/vectors and lines on surface
    """
    vectors = Vectors()
    n_cells = indices.shape[0]

    # add default fibres
    vectors.fibres = np.tile([1, 0, 0], (n_cells, 1))
    signal_maps = surface_data.get('signalMaps')
    if not signal_maps:
        return vectors

    # retrieve linear_connections indices
    lin_conns = signal_maps.get('linear_connections')
    vectors.linear_connections = lin_conns
    if isinstance(lin_conns, dict):
        vectors.linear_connections = lin_conns.get('value')

    # retrieve linear_connections region associated
    lin_conns_region = signal_maps.get('linear_connection_regions')
    vectors.linear_connection_regions = lin_conns_region
    if isinstance(lin_conns_region, dict):
        vectors.linear_connection_regions = lin_conns_region.get('value')

    # retrieve fibres
    _fibres = signal_maps.get('fibres')
    if isinstance(_fibres, dict):
        _fibres = _fibres.get('value')

    if _fibres is not None:
        # Ensure fibres match elem/face data size
        # Exclude fibre data associated with linear regions
        vectors.fibres = _fibres[0:n_cells]

    # extract custom fields (based on propSettings > type : 'vector')
    if signal_maps:
        for vector_name, vector_dict in signal_maps.items():

            if not isinstance(vector_dict, dict):
                # If not dictionary, ignore
                continue

            surface_props = vector_dict.get('propSettings', None)

            if surface_props.get('type') != 'vector':
                # If not vector, ignore
                continue

            vector_values = vector_dict.get('value', None)
            if isinstance(vector_values, np.ndarray):
                vector_values = None if vector_values.size == 0 else vector_values

            vectors[vector_name] = vector_values

    return vectors
