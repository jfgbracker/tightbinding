# A tight-binding Hamiltonian constructor

This package is designed to help the creation of tight-binding Hamiltonian matrices depending on multiple parameters. It also provides interactive plotting functions that can be run in a jupyter notebook to facilitate the study of band and eigenvector structures.

Hamiltonians are stored as [xarray](https://docs.xarray.dev/) `DataArray`s whose named
dimensions are the parameters of the model, so an entire parameter sweep is built,
diagonalised and sliced in one go.

Requires Python 3.11 or newer.

## Installation

First download the repository and extract it where you want. Then, run in your python environment

```bash
pip install path/to/package/tightbinding
```

or

```bash
pip install -e path/to/package/tightbinding
```

if you want to be able to modify it in place.

## Getting started

Once you have installed the package, open in a jupyter viewer the [Getting Started](docs/GettingStarted.ipynb) notebook.

```python
import numpy as np
from tightbinding import Orbital, Site, Unitcell, Lattice, Hopping, Hamiltonianbuilder

a = 2.4
uc = Unitcell("honeycomb", [[1.5 * a, -3**0.5 / 2 * a], [1.5 * a, 3**0.5 / 2 * a]])
uc.add_site([Site("A", [-a / 2, 0], Orbital("s")),
             Site("B", [a / 2, 0], Orbital("s"))])

lattice = Lattice("honeycomb", uc, periodicity=(True, True))
lattice.add_unitcell((0, 0), update=True)

H = Hamiltonianbuilder(lattice, {"t": -1.0, "kx": 0.0, "ky": 0.0}, ["kx", "ky"])
H.add_couplings([Hopping(lattice, "t", "A_s", "B_s", d)
                 for d in [(0, 0), (-1, 0), (0, -1)]])
H.build()
energies, states = H.eigh()
```

### Sampling the Brillouin zone

Take the reciprocal basis from the unit cell rather than deriving it by hand — a wrong
basis silently samples something that is not the Brillouin zone, and integrated
quantities such as Chern numbers stop being quantised:

```python
b1, b2 = uc.reciprocal_vectors()          # rows satisfy a_i . b_j = 2 pi delta_ij
frac = np.linspace(0, 1, n, endpoint=False)   # do not duplicate the zone edge
```

Pass `periodic=True` to `berry_curvature` when the grid tiles the zone this way.

### Plotting in notebooks

Importing the package no longer changes plotly's global configuration. If figures do
not display, call once at the top of your notebook:

```python
from tightbinding.plotting import configure_notebook_renderer
configure_notebook_renderer()
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The suite checks the assembled Hamiltonians against analytic results — the 1D chain
cosine band, the SSH spectrum and its gap, graphene's bandwidth and Dirac cone, the
quantised Haldane Chern number, and the agreement between the real-space and Bloch
assembly paths — alongside unit tests for the geometry and plotting layers.
