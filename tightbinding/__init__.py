"""A lattice-based tight-binding Hamiltonian constructor.

Lattices are built as a nested structure of :class:`Orbital`, :class:`Site`,
:class:`Unitcell` and :class:`Lattice` objects. :class:`Hamiltonianbuilder` then
assembles the Hamiltonian as an :class:`xarray.DataArray` whose named dimensions
are the parameters of the model, so a whole parameter sweep is diagonalised and
sliced at once.

    >>> from tightbinding import Orbital, Site, Unitcell, Lattice
    >>> uc = Unitcell("square", [[1, 0], [0, 1]])
    >>> uc.add_site(Site("A", [0, 0], Orbital("s")))
"""

from tightbinding.geometry import (
    Lattice,
    Orbital,
    Site,
    Unitcell,
    arraytostring,
    make_sliders,
    stringtoarray,
)
from tightbinding.hamiltonian import (
    Energy,
    Hamiltonianbuilder,
    Hopping,
    check_valid_coord,
)
from tightbinding.topology import berry_curvature, braket, euler_curvature, euler_patch

__version__ = "0.1.0"

#: Alias kept because the documentation and notebooks use the CamelCase spelling.
HamiltonianBuilder = Hamiltonianbuilder

__all__ = [
    "Energy",
    "HamiltonianBuilder",
    "Hamiltonianbuilder",
    "Hopping",
    "Lattice",
    "Orbital",
    "Site",
    "Unitcell",
    "arraytostring",
    "berry_curvature",
    "braket",
    "check_valid_coord",
    "euler_curvature",
    "euler_patch",
    "make_sliders",
    "stringtoarray",
    "__version__",
]
