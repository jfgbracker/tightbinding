"""Tests for the lattice geometry objects."""

import numpy as np
import pytest

from conftest import chain_unitcell, cubic_unitcell, honeycomb_unitcell
from tightbinding.geometry import (
    Lattice,
    Orbital,
    Site,
    Unitcell,
    arraytostring,
    stringtoarray,
)


# --------------------------------------------------------------------------
# name <-> array round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("array", [[0], [0, 0], [1, -2], [-3, 4, -5], [10, 200]])
def test_unitcell_name_round_trip(array):
    assert list(stringtoarray(arraytostring(np.array(array)))) == array


# --------------------------------------------------------------------------
# Orbital / Site
# --------------------------------------------------------------------------


def test_site_add_and_remove_orbitals():
    site = Site("A", [0.0, 0.0], [Orbital("s"), Orbital("px")])
    assert site.num_orbitals() == 2
    assert set(site.orbital_index) == {"s", "px"}

    site.add_orbital(Orbital("py"))
    assert site.num_orbitals() == 3
    assert sorted(site.orbital_index.values()) == [0, 1, 2]

    site.remove_orbital("px")
    assert site.num_orbitals() == 2
    assert "px" not in site.orbitals
    assert sorted(site.orbital_index.values()) == [0, 1]


def test_site_default_orbital_list_is_not_shared():
    """A mutable default argument would leak orbitals between sites."""
    first = Site("A", [0.0, 0.0])
    first.add_orbital(Orbital("s"))
    second = Site("B", [1.0, 0.0])
    assert second.num_orbitals() == 0


def test_site_rejects_non_orbitals():
    with pytest.raises(TypeError):
        Site("A", [0.0, 0.0], ["s"])


# --------------------------------------------------------------------------
# Unitcell
# --------------------------------------------------------------------------


def test_unitcell_counts_and_index():
    uc = honeycomb_unitcell()
    assert uc.dimension == 2
    assert uc.num_sites() == 2
    assert uc.num_orbitals() == 2
    assert uc.orbitals_index == {"A_s": 0, "B_s": 1}


def test_unitcell_rejects_non_square_unitvecs():
    with pytest.raises(ValueError):
        Unitcell("bad", np.array([[1.0, 0.0]]))


def test_unitcell_rejects_site_of_wrong_dimension():
    uc = Unitcell("c", np.eye(2))
    with pytest.raises(ValueError):
        uc.add_site(Site("A", [0.0, 0.0, 0.0], [Orbital("s")]))


def test_unitcell_stores_copies_of_sites():
    site = Site("A", [0.0, 0.0], [Orbital("s")])
    uc = Unitcell("c", np.eye(2))
    uc.add_site(site)
    site.add_orbital(Orbital("px"))
    assert uc.sites["A"].num_orbitals() == 1, "unit cell aliased the caller's Site"


def test_unitcell_combine_doubles_the_cell():
    uc = honeycomb_unitcell()
    a1, a2 = uc.unitvecs
    combined = Unitcell.combine([uc, uc], [a1], np.array([a1 + a2, a2 - a1]))
    assert combined.num_sites() == 4
    assert combined.num_orbitals() == 4
    assert set(combined.sites) == {"A", "B", "A-1", "B-1"}


def test_unitcell_combine_rejects_mismatched_lengths():
    uc = honeycomb_unitcell()
    with pytest.raises(ValueError):
        Unitcell.combine([uc, uc], [], uc.unitvecs)


def test_unitcell_reciprocal_vectors():
    uc = honeycomb_unitcell()
    products = np.asarray(uc.unitvecs) @ uc.reciprocal_vectors().T
    assert np.allclose(products, 2 * np.pi * np.eye(2))


# --------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------


def test_lattice_default_periodicity_matches_dimension():
    """The default must not assume a 3D lattice."""
    for uc in (chain_unitcell(), honeycomb_unitcell(), cubic_unitcell()):
        lat = Lattice(uc.name, uc)
        assert len(lat.periodicity) == uc.dimension
        assert not any(lat.periodicity)


def test_lattice_rejects_periodicity_of_wrong_length():
    with pytest.raises(ValueError):
        Lattice("hc", honeycomb_unitcell(), (True, True, True))


def test_create_rectangle_lattice_counts():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((3, 4))
    assert lat.num_unitcells() == 12
    assert lat.num_sites() == 24
    assert lat.num_orbitals() == 24
    assert sorted(lat.orbitals_index.values()) == list(range(24))
    assert lat.reverse_orbital_index[0] in lat.orbitals_index


def test_lattice_bounds_track_added_cells():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    for pos in [(-1, 0), (0, 0), (2, 3)]:
        lat.add_unitcell(pos)
    lat.define_index()
    assert list(lat.mincoords) == [-1, 0]
    assert list(lat.maxcoords) == [2, 3]
    assert list(lat.range) == [4, 4]


def test_find_closest_unitcell():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((3, 3))
    for target in ["[0 0]", "[1 1]", "[2 2]"]:
        assert lat.find_closest_uc(lat.unitcells[target].position) == target


def test_remove_unitcell_by_cartesian_position():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((3, 3))
    target = lat.unitcells["[1 1]"].position.copy()
    lat.remove_unit_cell(target, coordsys="xy")
    assert "[1 1]" not in lat.unitcells
    assert lat.num_unitcells() == 8


def test_remove_site_by_cartesian_position():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((3, 3))
    target = lat.unitcells["[1 1]"].position.copy()
    lat.remove_site(target, "A", coordsys="xy")
    assert "A" not in lat.unitcells["[1 1]"].sites
    assert lat.num_orbitals() == 17


def test_removing_every_site_removes_the_unitcell():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((2, 2))
    lat.remove_site((0, 0), "A")
    lat.remove_site((0, 0), "B")
    assert "[0 0]" not in lat.unitcells


def test_remove_orbital_from_lattice():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((2, 2))
    lat.remove_orbital((0, 0), "A", "s")
    assert lat.num_orbitals() == 7
    assert "A" not in lat.unitcells["[0 0]"].sites


def test_evaluate_on_lattice_without_return_annotation():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((2, 2))
    field = lat.evaluate_on_lattice(lambda x, y: x + y)
    assert field.shape == (lat.num_orbitals(),)

    positions = []
    for kuc, uc in lat.unitcells.items():
        for site in uc.sites.values():
            positions.append(uc.position + site.position)
    assert np.allclose(sorted(field.values), sorted(p[0] + p[1] for p in positions))


def test_evaluate_on_lattice_rejects_wrong_arity():
    lat = Lattice("hc", honeycomb_unitcell(), (False, False))
    lat.create_rectangle_lattice((2, 2))
    with pytest.raises(ValueError):
        lat.evaluate_on_lattice(lambda x: x)
