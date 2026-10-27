"""Shared fixtures and model factories for the tightbinding test suite.

The factories here build the standard textbook models that the physics tests
check against analytic results: a 1D chain, the SSH chain, graphene and the
Haldane model.

Two k-space parametrisations are offered:

``*_bloch``       flat ``kpt`` dimension, for tests that only look at energies;
``*_bloch_grid``  separate ``ka1``/``ka2`` dimensions, needed by
                  :func:`tightbinding.topology.berry_curvature`, which requires
                  two distinct reciprocal dimensions.
"""

import numpy as np
import pytest

from tightbinding.geometry import Lattice, Orbital, Site, Unitcell
from tightbinding.hamiltonian import Energy, Hamiltonianbuilder, Hopping

A_LAT = 2.4  # lattice constant used throughout the tests


def reciprocal_vectors(unitvecs):
    """Reference implementation of the reciprocal basis, B = 2*pi*inv(A).T."""
    return 2 * np.pi * np.linalg.inv(np.asarray(unitvecs, dtype=float)).T


# --------------------------------------------------------------------------
# unit cells
# --------------------------------------------------------------------------


def chain_unitcell(a=A_LAT):
    """A one dimensional chain with a single s orbital per cell."""
    uc = Unitcell("chain", np.array([[a]]))
    uc.add_site(Site("A", [0.0], [Orbital("s")]))
    return uc


def ssh_unitcell(a=A_LAT):
    """A 1D two-site cell (the SSH chain)."""
    uc = Unitcell("ssh", np.array([[a]]))
    uc.add_site([Site("A", [-a / 4], [Orbital("s")]),
                 Site("B", [a / 4], [Orbital("s")])])
    return uc


def honeycomb_unitcell(a=A_LAT):
    """The two-site honeycomb (graphene) unit cell."""
    a1 = np.array([1.5 * a, -np.sqrt(3) / 2 * a])
    a2 = np.array([1.5 * a, np.sqrt(3) / 2 * a])
    uc = Unitcell("honeycomb", np.array([a1, a2]))
    uc.add_site([Site("A", [-a / 2, 0.0], [Orbital("s")]),
                 Site("B", [a / 2, 0.0], [Orbital("s")])])
    return uc


def cubic_unitcell(a=1.0):
    """A two-site 3D cell, used to check that 3D lattices build at all."""
    uc = Unitcell("cubic", a * np.eye(3))
    uc.add_site([Site("A", [0.0, 0.0, 0.0], [Orbital("s")]),
                 Site("B", [a / 2, a / 2, a / 2], [Orbital("s")])])
    return uc


@pytest.fixture
def honeycomb_uc():
    return honeycomb_unitcell()


@pytest.fixture
def honeycomb_lattice():
    """A 4x4 open-boundary honeycomb lattice."""
    lat = Lattice("honeycomb", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((4, 4))
    return lat


# --------------------------------------------------------------------------
# k-space helpers
# --------------------------------------------------------------------------


def bloch_lattice(unitcell):
    """A single-unit-cell lattice with periodic boundaries: a Bloch Hamiltonian."""
    dim = unitcell.dimension
    lat = Lattice(unitcell.name, unitcell, (True,) * dim)
    lat.add_unitcell((0,) * dim, update=True)
    return lat


def k_points(unitvecs, n, endpoint=False):
    """Flat array of k points, shape ``(n**dim, dim)``, covering one BZ period."""
    bvecs = reciprocal_vectors(unitvecs)
    axes = [np.linspace(0, 1, n, endpoint=endpoint) for _ in range(len(bvecs))]
    mesh = np.meshgrid(*axes, indexing="ij")
    frac = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    return frac @ bvecs


def attach_k(builder, kcomponents, dims):
    """Attach cartesian k components as multidimensional coords kx/ky/kz."""
    for name, values in zip(("kx", "ky", "kz"), kcomponents):
        builder.add_multidimensional_coord(name, values, dims)


def _flat_bloch(unitcell, params, couplings, n, onsite=None):
    lat = bloch_lattice(unitcell)
    kpts = k_points(unitcell.unitvecs, n)
    recoords = ["kx", "ky", "kz"][: unitcell.dimension]
    params = {**params, "kpt": np.arange(len(kpts))}
    H = Hamiltonianbuilder(lat, params, recoords)
    attach_k(H, [kpts[:, i] for i in range(kpts.shape[1])], ("kpt",))
    H.add_couplings([Hopping(lat, expr, o1, o2, d) for expr, o1, o2, d in couplings])
    if onsite is not None:
        H.set_on_site_energies(onsite(lat))
    H.build()
    return H, kpts


def _grid_bloch(unitcell, params, couplings, n, onsite=None, endpoint=False):
    """2D Bloch Hamiltonian on separate ka1/ka2 dimensions (for Berry curvature)."""
    lat = bloch_lattice(unitcell)
    bvecs = reciprocal_vectors(unitcell.unitvecs)
    frac = np.linspace(0, 1, n, endpoint=endpoint)
    F1, F2 = np.meshgrid(frac, frac, indexing="ij")
    kx = F1 * bvecs[0][0] + F2 * bvecs[1][0]
    ky = F1 * bvecs[0][1] + F2 * bvecs[1][1]
    params = {**params, "ka1": frac, "ka2": frac}
    H = Hamiltonianbuilder(lat, params, ["kx", "ky"])
    attach_k(H, [kx, ky], ("ka1", "ka2"))
    H.add_couplings([Hopping(lat, expr, o1, o2, d) for expr, o1, o2, d in couplings])
    if onsite is not None:
        H.set_on_site_energies(onsite(lat))
    H.build()
    return H


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

GRAPHENE_NN = [(0, 0), (-1, 0), (0, -1)]
HALDANE_NNN = [(1, 0), (0, -1), (-1, 1)]


def chain_bloch(t=-1.0, n=64, a=A_LAT):
    """1D chain. Analytic bands: E(k) = 2 t cos(k a)."""
    return _flat_bloch(chain_unitcell(a), {"t": t},
                       [("t", "A_s", "A_s", (1,))], n)


def ssh_bloch(v=-1.0, w=-0.4, n=64, a=A_LAT):
    """SSH chain. Analytic bands: E(k) = +-|v + w exp(-i k a)|."""
    return _flat_bloch(ssh_unitcell(a), {"v": v, "w": w},
                       [("v", "A_s", "B_s", (0,)), ("w", "B_s", "A_s", (1,))], n)


def graphene_bloch(t=-1.0, n=60, a=A_LAT):
    """Nearest-neighbour graphene. Bandwidth 3|t|, gapless Dirac points at K."""
    return _flat_bloch(honeycomb_unitcell(a), {"t": t},
                       [("t", "A_s", "B_s", d) for d in GRAPHENE_NN], n)


def haldane_couplings(t2=0.15):
    return ([("t", "A_s", "B_s", d) for d in GRAPHENE_NN]
            + [(f"{t2}j", "A_s", "A_s", d) for d in HALDANE_NNN]
            + [(f"{-t2}j", "B_s", "B_s", d) for d in HALDANE_NNN])


def haldane_grid(t=-1.0, t2=0.15, mass=0.0, n=48, a=A_LAT, endpoint=False):
    """Haldane model on a 2D k grid. Lower-band Chern number is -1 for mass=0.

    A staggered ``mass`` larger than ``3*sqrt(3)*t2`` drives it into the
    trivial phase.
    """
    onsite = None
    if mass:
        def onsite(lat):
            e = Energy(lat)
            e.set_energy(f"{mass}", nsite="A")
            e.set_energy(f"{-mass}", nsite="B")
            return e

    return _grid_bloch(honeycomb_unitcell(a), {"t": t}, haldane_couplings(t2),
                       n, onsite=onsite, endpoint=endpoint)
