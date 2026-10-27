"""Tests for the Hamiltonian assembly layer: Hopping, Energy, Hamiltonianbuilder."""

import numpy as np
import pytest
import xarray as xr

from conftest import (
    GRAPHENE_NN,
    bloch_lattice,
    chain_unitcell,
    cubic_unitcell,
    graphene_bloch,
    honeycomb_unitcell,
)
from tightbinding.geometry import Lattice, Orbital, Site, Unitcell
from tightbinding.hamiltonian import (
    Energy,
    Hamiltonianbuilder,
    Hopping,
    check_valid_coord,
)


def open_honeycomb(dims=(3, 3)):
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice(dims)
    return lat


# --------------------------------------------------------------------------
# check_valid_coord
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.arange(3), True),
        (xr.DataArray(np.arange(3)), True),
        (1.0, False),
        (-1, False),
        (np.zeros((2, 2)), False),
        ("string", False),
        (None, False),
    ],
)
def test_check_valid_coord(value, expected):
    assert check_valid_coord(value) is expected


# --------------------------------------------------------------------------
# Hopping
# --------------------------------------------------------------------------


def test_hopping_distance_for_nearest_neighbours():
    """All three graphene NN bonds have length equal to the bond length a."""
    uc = honeycomb_unitcell()
    lat = bloch_lattice(uc)
    for disp in GRAPHENE_NN:
        hop = Hopping(lat, "t", "A_s", "B_s", disp)
        assert np.linalg.norm(hop.std_dist) == pytest.approx(2.4, abs=1e-12)


def test_hopping_accepts_multicharacter_site_names():
    """Site names must not be truncated to their first character."""
    a = 2.4
    uc = Unitcell("hc", np.array([[1.5 * a, -np.sqrt(3) / 2 * a],
                                  [1.5 * a, np.sqrt(3) / 2 * a]]))
    uc.add_site([Site("Aup", [-a / 2, 0.0], [Orbital("s")]),
                 Site("Bup", [a / 2, 0.0], [Orbital("s")])])
    lat = bloch_lattice(uc)
    hop = Hopping(lat, "t", "Aup_s", "Bup_s", (0, 0))
    assert np.allclose(hop.std_dist, [a, 0.0])


def test_hopping_end_unit_cell_wraps_only_when_periodic():
    lat = open_honeycomb((3, 3))
    hop = Hopping(lat, "t", "A_s", "B_s", (1, 0))
    _, exists, allowed = hop.end_unit_cell("[2 0]")
    assert exists and not allowed, "open boundary must forbid the wrap"

    lat.periodicity = (True, True)
    _, exists, allowed = hop.end_unit_cell("[2 0]")
    assert exists and allowed
    assert hop.end_unit_cell("[2 0]")[0] == "[0 0]"


def test_hopping_reports_missing_orbitals():
    lat = open_honeycomb((2, 2))
    lat.remove_site((1, 1), "B")
    hop = Hopping(lat, "t", "A_s", "B_s", (0, 0))
    assert hop.end_unit_cell("[1 1]")[1] == 0


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------


def test_energy_starts_at_zero_for_every_orbital():
    lat = open_honeycomb((2, 2))
    energy = Energy(lat)
    assert set(energy.onsite_energies) == set(lat.orbitals_index)
    assert set(energy.onsite_energies.values()) == {"0"}


def test_set_energy_without_filters_affects_every_orbital():
    """A bare set_energy() must set all orbitals, not silently do nothing."""
    lat = open_honeycomb((2, 2))
    energy = Energy(lat)
    energy.set_energy("5.0")
    assert set(energy.onsite_energies) == set(lat.orbitals_index)
    assert set(energy.onsite_energies.values()) == {"5.0"}

    H = Hamiltonianbuilder(lat, {})
    H.set_on_site_energies(energy)
    H.build()
    diagonal = np.diagonal(H.Hamiltonian.values)
    assert np.allclose(diagonal, 5.0)


def test_set_energy_by_site():
    lat = open_honeycomb((2, 2))
    energy = Energy(lat)
    energy.set_energy("-1.5", nsite="A")
    energy.set_energy("2.5", nsite="B")
    for key, expr in energy.onsite_energies.items():
        assert expr == ("-1.5" if "_A_" in key else "2.5")


def test_set_energy_by_orbital_only():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.unitcell.sites["A"].add_orbital(Orbital("px"))
    lat = Lattice("hc", lat.unitcell, (False, False))
    lat.create_rectangle_lattice((2, 2))

    energy = Energy(lat)
    energy.set_energy("3.0", norbital="px")
    assert all(expr == "3.0" for key, expr in energy.onsite_energies.items()
               if key.endswith("_px"))
    assert all(expr == "0" for key, expr in energy.onsite_energies.items()
               if key.endswith("_s"))


def test_set_energy_by_unitcell():
    lat = open_honeycomb((2, 2))
    energy = Energy(lat)
    energy.set_energy("7.0", nunitcell="[0 0]")
    for key, expr in energy.onsite_energies.items():
        assert expr == ("7.0" if key.startswith("[0 0]") else "0")


def test_set_energy_with_condition_on_index():
    lat = open_honeycomb((3, 3))
    energy = Energy(lat)
    energy.set_energy("1.0", condition="index[0] == 0")
    affected = {k for k, v in energy.onsite_energies.items() if v == "1.0"}
    assert affected == {k for k in lat.orbitals_index if k.startswith("[0 ")}


def test_set_energy_with_condition_on_position():
    lat = open_honeycomb((3, 3))
    energy = Energy(lat)
    energy.set_energy("1.0", condition="position[0] > 0")
    assert 0 < len({k for k, v in energy.onsite_energies.items() if v == "1.0"}) \
        < len(lat.orbitals_index)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"norbital": "s"},
        {"nsite": "A"},
        {"nsite": "A", "norbital": "s"},
        {"nunitcell": "[0 0]"},
        {"nunitcell": "[0 0]", "nsite": "A"},
        {"nunitcell": "[0 0]", "norbital": "s"},
        {"nunitcell": "[0 0]", "nsite": "A", "norbital": "s"},
    ],
)
def test_add_energy_accumulates_for_every_selector(kwargs):
    """Every selector combination must run and add to the existing expression."""
    lat = open_honeycomb((2, 2))
    energy = Energy(lat)
    energy.set_energy("1.0")
    energy.add_energy("2.0", **kwargs)

    changed = {k for k, v in energy.onsite_energies.items() if v != "1.0"}
    assert changed, f"add_energy{kwargs} affected nothing"

    H = Hamiltonianbuilder(lat, {})
    H.set_on_site_energies(energy)
    H.build()
    diagonal = np.diagonal(H.Hamiltonian.values).real
    assert set(np.round(diagonal, 9)) <= {1.0, 3.0}
    assert 3.0 in set(np.round(diagonal, 9))


# --------------------------------------------------------------------------
# Hamiltonianbuilder
# --------------------------------------------------------------------------


def test_builder_array_shape_follows_parameters():
    lat = open_honeycomb((2, 2))
    params = {"t": -1.0, "eps": np.linspace(-1, 1, 5), "delta": np.linspace(0, 1, 3)}
    H = Hamiltonianbuilder(lat, params)
    assert H.Hamiltonian.dims == ("eps", "delta", "i", "j")
    assert H.Hamiltonian.shape == (5, 3, 8, 8)


def test_builder_real_flag_gives_real_dtype():
    lat = open_honeycomb((2, 2))
    H = Hamiltonianbuilder(lat, {"t": -1.0}, real=True)
    assert H.Hamiltonian.dtype == np.float64
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()
    assert H.Hamiltonian.dtype == np.float64


def test_build_is_idempotent():
    """Calling build() twice must not double the matrix elements."""
    H, _ = graphene_bloch(n=8)
    first = H.Hamiltonian.values.copy()
    H.build()
    assert np.allclose(first, H.Hamiltonian.values)


def test_build_preserves_multidimensional_coords():
    H, _ = graphene_bloch(n=8)
    assert "kx" in H.Hamiltonian.coords
    assert "ky" in H.Hamiltonian.coords
    assert H.Hamiltonian.coords["kx"].dims == ("kpt",)


def test_add_parameters_extends_the_array():
    lat = open_honeycomb((2, 2))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    assert H.Hamiltonian.dims == ("i", "j")
    H.add_parameters({"eps": np.linspace(-1, 1, 4)})
    assert H.Hamiltonian.dims == ("eps", "i", "j")


def test_couplings_depend_on_parameters():
    lat = open_honeycomb((2, 2))
    eps = np.linspace(-1, 1, 5)
    H = Hamiltonianbuilder(lat, {"t": -1.0, "eps": eps})
    energy = Energy(lat)
    energy.set_energy("eps", nsite="A")
    H.set_on_site_energies(energy)
    H.add_couplings([Hopping(lat, "t * eps", "A_s", "B_s", (0, 0))])
    H.build()

    index_a = lat.orbitals_index["[0 0]_A_s"]
    index_b = lat.orbitals_index["[0 0]_B_s"]
    assert np.allclose(H.Hamiltonian.values[:, index_a, index_a].real, eps)
    assert np.allclose(H.Hamiltonian.values[:, index_a, index_b].real, -eps)


def test_open_boundaries_have_fewer_bonds_than_periodic():
    def bond_count(periodicity, extra):
        lat = Lattice("hc", honeycomb_unitcell(), periodicity)
        lat.create_rectangle_lattice((3, 3))
        H = Hamiltonianbuilder(lat, {"t": -1.0, **extra},
                               ["kx", "ky"] if any(periodicity) else [])
        H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
        H.build()
        return np.count_nonzero(H.Hamiltonian.values)

    open_bonds = bond_count((False, False), {})
    periodic_bonds = bond_count((True, True), {"kx": 0.0, "ky": 0.0})
    assert periodic_bonds > open_bonds
    assert periodic_bonds == 2 * 3 * 9  # 3 bonds per cell, both directions


def test_basis_exchange_preserves_the_spectrum():
    H, _ = graphene_bloch(n=8)
    before, _ = H.eigh()
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    H.basis_exchange(rotation)
    after, _ = H.eigh()
    assert np.allclose(np.sort(before.values, axis=-1),
                       np.sort(after.values, axis=-1), atol=1e-10)


def test_basis_exchange_rejects_bad_shapes():
    H, _ = graphene_bloch(n=4)
    with pytest.raises(ValueError):
        H.basis_exchange(np.zeros((2, 3)))


def test_eigh_returns_orthonormal_eigenvectors():
    H, _ = graphene_bloch(n=8)
    _, eigve = H.eigh()
    vectors = eigve.transpose("kpt", "component", "band").values
    overlap = np.conj(np.swapaxes(vectors, -1, -2)) @ vectors
    assert np.allclose(overlap, np.eye(2)[None, :, :], atol=1e-10)


def test_eigh_solves_the_eigenproblem():
    H, _ = graphene_bloch(n=8)
    eigva, eigve = H.eigh()
    matrices = H.Hamiltonian.transpose("kpt", "i", "j").values
    vectors = eigve.transpose("kpt", "component", "band").values
    values = eigva.transpose("kpt", "band").values
    assert np.allclose(matrices @ vectors, vectors * values[:, None, :], atol=1e-10)


def test_eigh_fixes_the_gauge_on_the_first_component():
    H, _ = graphene_bloch(n=16)
    _, eigve = H.eigh()
    first = eigve.sel(component=0).values
    assert np.allclose(first.imag, 0.0, atol=1e-12)
    assert np.all(first.real >= -1e-12)


def test_eigh_parallel_matches_eigh():
    H, _ = graphene_bloch(n=8)
    serial, _ = H.eigh()
    parallel, _ = H.eigh_parallel()
    assert np.allclose(np.sort(serial.values, axis=-1),
                       np.sort(parallel.transpose(*serial.dims).values, axis=-1),
                       atol=1e-10)


def test_eigh_parallel_keeps_real_hamiltonians_real():
    lat = open_honeycomb((3, 3))
    H = Hamiltonianbuilder(lat, {"t": -1.0, "eps": np.linspace(-1, 1, 4)}, real=True)
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()
    eigva, eigve = H.eigh_parallel()
    assert not np.iscomplexobj(eigve.values)


def test_set_type_and_make_real():
    lat = open_honeycomb((2, 2))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    assert H.Hamiltonian.dtype == np.complex128
    H.make_real()
    assert H.Hamiltonian.dtype == np.float64
    H.set_type(np.complex128)
    assert H.Hamiltonian.dtype == np.complex128


# --------------------------------------------------------------------------
# dimensionality coverage
# --------------------------------------------------------------------------


def test_one_dimensional_lattice_builds():
    lat = Lattice("chain", chain_unitcell(), (False,))
    lat.create_rectangle_lattice((5,))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    H.add_couplings([Hopping(lat, "t", "A_s", "A_s", (1,))])
    H.build()
    assert H.Hamiltonian.shape == (5, 5)
    assert np.count_nonzero(H.Hamiltonian.values) == 8  # 4 bonds, both directions


def test_three_dimensional_lattice_builds():
    uc = cubic_unitcell()
    lat = bloch_lattice(uc)
    kvals = np.linspace(0, 1, 4)
    H = Hamiltonianbuilder(
        lat, {"t": -1.0, "kx": kvals, "ky": kvals, "kz": kvals}, ["kx", "ky", "kz"]
    )
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", (0, 0, 0))])
    H.build()
    assert H.Hamiltonian.shape == (4, 4, 4, 2, 2)
    values = H.Hamiltonian.values
    assert np.allclose(values, np.conj(np.swapaxes(values, -1, -2)))


def test_hopping_into_a_cell_the_start_cell_does_not_share():
    """A vacancy must not break hoppings that end on a site missing upstream.

    The destination orbital has to be looked up in the end unit cell, not the
    starting one.
    """
    lat = open_honeycomb((3, 3))
    lat.remove_site((0, 0), "B")
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", (1, 0))])
    H.build()

    index_a = lat.orbitals_index["[0 0]_A_s"]
    index_b = lat.orbitals_index["[1 0]_B_s"]
    assert H.Hamiltonian.values[index_a, index_b] != 0


def test_orbital_angle_dependence_modulates_the_hopping():
    """A p-like orbital (one node) must flip sign with the bond direction."""
    uc = honeycomb_unitcell()
    uc.sites["A"].orbitals["s"].set_angle_dependance((1.0,))
    lat = bloch_lattice(uc)
    H = Hamiltonianbuilder(lat, {"t": -1.0, "kx": 0.0, "ky": 0.0}, ["kx", "ky"])
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()

    plain = Hamiltonianbuilder(bloch_lattice(honeycomb_unitcell()),
                               {"t": -1.0, "kx": 0.0, "ky": 0.0}, ["kx", "ky"])
    lat_plain = plain.lattice
    plain.add_couplings([Hopping(lat_plain, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    plain.build()

    assert not np.allclose(H.Hamiltonian.values, plain.Hamiltonian.values)
