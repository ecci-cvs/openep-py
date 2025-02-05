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
    fibres: np.ndarray = None
    divergence: np.ndarray = None
    linear_connections: np.ndarray = None
    linear_connection_regions: np.ndarray = None

    def __repr__(self):
        return f"vectors: {tuple(self.__dict__.keys())}"

    def __getitem__(self, arrow):
        try:
            return self.__dict__[arrow]
        except KeyError:
            raise ValueError(f"There is no vector '{arrow}'.")

    def __setitem__(self, arrow, value):
        if arrow not in self.__dict__.keys():
            raise ValueError(f"'{arrow}' is not a valid vector name.")
        self.__dict__[arrow] = value

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __contains__(self, arrow):
        return arrow in self.__dict__.keys()

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
    n_fibres = indices.shape[0]

    # add fibres
    default_fibres_data = np.tile([1, 0, 0], (n_fibres - 1, 1))

    if not surface_data.get('signalMaps'):
        vectors.fibres = default_fibres_data
        return vectors

    signal_props = surface_data.get('signalMaps')

    lin_conns = signal_props.get('linear_connections')
    if lin_conns is not None:
        if isinstance(lin_conns, dict):
            vectors.linear_connections = lin_conns.get('value')
        else:
            vectors.linear_connections = lin_conns

    lin_conns_region = signal_props.get('linear_connection_regions')
    if lin_conns_region is not None:
        if isinstance(lin_conns_region, dict):
            vectors.linear_connection_regions = lin_conns_region.get('value')
        else:
            vectors.linear_connection_regions = lin_conns_region

        n_fibres += len(vectors.linear_connection_regions)

    fibres = signal_props.get('fibres')
    if fibres is not None:
        if isinstance(fibres, dict):
            vectors.fibres = fibres.get('value')
        else:
            vectors.linear_connection_regions = lin_conns_region
    else:
        vectors.fibres = default_fibres_data

    return vectors
