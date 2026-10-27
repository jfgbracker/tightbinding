"""Physics benchmarks: every result here is checked against an analytic value.

These are the tests that pin down that the Hamiltonian is actually built
correctly; the rest of the suite mostly guards against regressions in the
plumbing.
"""

import numpy as np
import pytest

from conftest import (
    A_LAT,
    GRAPHENE_NN,
    bloch_lattice,
    chain_bloch,
    graphene_bloch,
    haldane_grid,
    honeycomb_unitcell,
    k_points,
    reciprocal_vectors,
    ssh_bloch,
)
from tightbinding.geometry import Lattice
from tightbinding.hamiltonian import Hamiltonianbuilder, Hopping
from tightbinding.topology import berry_curvature


def assert_hermitian(builder):
    values = builder.Hamiltonian.values
    assert np.allclose(values, np.conj(np.swapaxes(values, -1, -2))), (
        "Hamiltonian is not Hermitian"
    )


# --------------------------------------------------------------------------
# 1D chain: E(k) = 2 t cos(k a)
# --------------------------------------------------------------------------


def test_chain_bands_match_cosine():
    t = -1.0
    H, kpts = chain_bloch(t=t, n=64)
    assert_hermitian(H)
    energies, _ = H.eigh()
    expected = 2 * t * np.cos(kpts[:, 0] * A_LAT)
    assert np.allclose(np.sort(energies.values, axis=-1)[:, 0], expected, atol=1e-12)


def test_chain_bandwidth():
    H, _ = chain_bloch(t=-1.0, n=64)
    energies, _ = H.eigh()
    assert float(energies.max() - energies.min()) == pytest.approx(4.0, abs=1e-9)


# --------------------------------------------------------------------------
# SSH chain: E(k) = +-|v + w exp(-i k a)|
# --------------------------------------------------------------------------


@pytest.mark.parametrize("v, w", [(-1.0, -0.4), (-0.4, -1.0), (-1.0, -1.0)])
def test_ssh_bands_match_analytic(v, w):
    H, kpts = ssh_bloch(v=v, w=w, n=64)
    assert_hermitian(H)
    energies, _ = H.eigh()
    expected = np.abs(v + w * np.exp(-1j * kpts[:, 0] * A_LAT))
    computed = np.sort(energies.values, axis=-1)
    assert np.allclose(computed[:, 1], expected, atol=1e-12)
    assert np.allclose(computed[:, 0], -expected, atol=1e-12)


def test_ssh_gap_closes_at_dimerisation_transition():
    """The SSH gap is 2|v - w| and closes only when v == w.

    The minimum sits at the zone boundary ka = pi, so an even number of
    sampling points is needed to land on it exactly.
    """
    for v, w in [(-1.0, -0.4), (-1.0, -1.0), (-0.4, -1.0)]:
        H, _ = ssh_bloch(v=v, w=w, n=256)
        energies, _ = H.eigh()
        gap = float((energies.sel(band=1) - energies.sel(band=0)).min())
        assert gap == pytest.approx(2 * abs(v - w), abs=1e-9)


# --------------------------------------------------------------------------
# graphene
# --------------------------------------------------------------------------


def test_graphene_bandwidth_is_three_t():
    t = -1.0
    H, _ = graphene_bloch(t=t, n=60)
    assert_hermitian(H)
    energies, _ = H.eigh()
    assert float(abs(energies).max()) == pytest.approx(3 * abs(t), abs=1e-9)


def test_graphene_is_gapless_at_the_dirac_point():
    """K = (b1 - b2)/3 sits on the grid when n is a multiple of 3."""
    H, _ = graphene_bloch(t=-1.0, n=63)
    energies, _ = H.eigh()
    gap = energies.sel(band=1) - energies.sel(band=0)
    assert float(gap.min()) == pytest.approx(0.0, abs=1e-10)


def test_graphene_dirac_cone_is_linear():
    """Near K the dispersion is E = hbar v_F |q| with v_F = 3|t|a/2."""
    uc = honeycomb_unitcell()
    lat = bloch_lattice(uc)
    bvecs = reciprocal_vectors(uc.unitvecs)
    K = (bvecs[0] - bvecs[1]) / 3.0

    q = np.logspace(-4, -2, 12)
    kpts = K[None, :] + np.stack([q, np.zeros_like(q)], axis=-1)

    H = Hamiltonianbuilder(lat, {"t": -1.0, "kpt": np.arange(len(q))}, ["kx", "ky"])
    H.add_multidimensional_coord("kx", kpts[:, 0], ("kpt",))
    H.add_multidimensional_coord("ky", kpts[:, 1], ("kpt",))
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()

    energies, _ = H.eigh()
    upper = energies.sel(band=1).values
    v_fermi = 3 * abs(-1.0) * A_LAT / 2
    assert np.allclose(upper, v_fermi * q, rtol=2e-3)


@pytest.mark.parametrize("shift_index", [0, 1])
def test_graphene_spectrum_is_brillouin_zone_periodic(shift_index):
    """E(k + b_i) == E(k): the guard against wrong reciprocal vectors.

    The builder works in the atomic gauge (phases carry the intra-cell orbital
    offsets), so H itself is only periodic up to a diagonal gauge transform.
    The *spectrum* is strictly periodic, and that is what pins down the
    reciprocal basis.
    """
    uc = honeycomb_unitcell()
    lat = bloch_lattice(uc)
    bvecs = reciprocal_vectors(uc.unitvecs)

    base = np.array([[0.3, 0.11], [-0.7, 0.25], [1.3, -0.4]])
    kpts = np.concatenate([base, base + bvecs[shift_index]])

    H = Hamiltonianbuilder(lat, {"t": -1.0, "kpt": np.arange(len(kpts))}, ["kx", "ky"])
    H.add_multidimensional_coord("kx", kpts[:, 0], ("kpt",))
    H.add_multidimensional_coord("ky", kpts[:, 1], ("kpt",))
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()

    energies, _ = H.eigh()
    values = energies.values
    assert np.allclose(values[: len(base)], values[len(base):], atol=1e-10)


# --------------------------------------------------------------------------
# Haldane model: quantised Chern number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t2, expected", [(0.15, -1), (-0.15, 1)])
def test_haldane_chern_number_is_quantised(t2, expected):
    H = haldane_grid(t2=t2, n=48)
    assert_hermitian(H)
    _, eigve = H.eigh()
    berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
    chern = float(berry.sel(band=0).sum()) / 2 / np.pi
    assert chern == pytest.approx(expected, abs=1e-6)


def test_haldane_trivial_phase_has_zero_chern_number():
    """A staggered mass above 3*sqrt(3)*t2 destroys the Chern phase."""
    H = haldane_grid(t2=0.15, mass=1.2, n=48)
    _, eigve = H.eigh()
    berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
    chern = float(berry.sel(band=0).sum()) / 2 / np.pi
    assert chern == pytest.approx(0.0, abs=1e-6)


def test_chern_number_is_independent_of_grid_resolution():
    for n in (24, 36, 48):
        H = haldane_grid(t2=0.15, n=n)
        _, eigve = H.eigh()
        berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
        chern = float(berry.sel(band=0).sum()) / 2 / np.pi
        assert chern == pytest.approx(-1, abs=1e-6), f"failed at n={n}"


# --------------------------------------------------------------------------
# real space <-> Bloch consistency
# --------------------------------------------------------------------------


def test_supercell_spectrum_equals_folded_bloch_bands():
    """An N x N periodic supercell at zero twist folds the Bloch bands.

    This ties the real-space assembly path (many unit cells, modular wrapping
    at the boundary) to the Bloch path (one cell, phase factors), which are
    otherwise almost independent pieces of code.
    """
    n_cells = 4
    uc = honeycomb_unitcell()

    supercell = Lattice("honeycomb", uc, (True, True))
    supercell.create_rectangle_lattice((n_cells, n_cells))
    Hs = Hamiltonianbuilder(supercell, {"t": -1.0, "kx": 0.0, "ky": 0.0}, ["kx", "ky"])
    Hs.add_couplings([Hopping(supercell, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    Hs.build()
    assert_hermitian(Hs)
    real_space, _ = Hs.eigh()

    kpts = k_points(uc.unitvecs, n_cells)
    lat = bloch_lattice(uc)
    Hb = Hamiltonianbuilder(lat, {"t": -1.0, "kpt": np.arange(len(kpts))}, ["kx", "ky"])
    Hb.add_multidimensional_coord("kx", kpts[:, 0], ("kpt",))
    Hb.add_multidimensional_coord("ky", kpts[:, 1], ("kpt",))
    Hb.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    Hb.build()
    bloch, _ = Hb.eigh()

    assert np.allclose(np.sort(real_space.values.reshape(-1)),
                       np.sort(bloch.values.reshape(-1)), atol=1e-10)


def test_open_chain_matches_analytic_particle_in_a_box():
    """An open N-site chain has E_m = 2 t cos(m pi / (N + 1))."""
    from conftest import chain_unitcell

    n_sites = 12
    t = -1.0
    lat = Lattice("chain", chain_unitcell(), (False,))
    lat.create_rectangle_lattice((n_sites,))
    H = Hamiltonianbuilder(lat, {"t": t})
    H.add_couplings([Hopping(lat, "t", "A_s", "A_s", (1,))])
    H.build()
    assert_hermitian(H)

    energies, _ = H.eigh()
    m = np.arange(1, n_sites + 1)
    expected = np.sort(2 * t * np.cos(m * np.pi / (n_sites + 1)))
    assert np.allclose(np.sort(energies.values), expected, atol=1e-10)


def test_lattice_origin_offset_does_not_change_the_spectrum():
    """Cells indexed -2..1 must give the same ring as cells indexed 0..3.

    Exercises the ``mincoords`` bookkeeping in the periodic wrapping.
    """
    from conftest import chain_unitcell

    t = -1.0
    spectra = []
    for offsets in ([0, 1, 2, 3], [-2, -1, 0, 1]):
        lat = Lattice("chain", chain_unitcell(), (True,))
        for i in offsets:
            lat.add_unitcell((i,))
        lat.define_index()
        H = Hamiltonianbuilder(lat, {"t": t, "kx": 0.0}, ["kx"])
        H.add_couplings([Hopping(lat, "t", "A_s", "A_s", (1,))])
        H.build()
        assert_hermitian(H)
        energies, _ = H.eigh()
        spectra.append(np.sort(energies.values))

    assert np.allclose(spectra[0], spectra[1], atol=1e-10)
