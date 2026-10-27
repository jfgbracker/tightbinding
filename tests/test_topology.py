"""Tests for the Berry-phase machinery in :mod:`tightbinding.topology`.

The Euler-invariant functions are deliberately not covered here.
"""

import numpy as np
import pytest
import xarray as xr

from conftest import haldane_grid
from tightbinding.topology import berry_curvature, braket


# --------------------------------------------------------------------------
# braket
# --------------------------------------------------------------------------


def test_braket_is_conjugate_linear_in_the_first_argument():
    rng = np.random.default_rng(0)
    a = xr.DataArray(rng.normal(size=4) + 1j * rng.normal(size=4), dims="component")
    b = xr.DataArray(rng.normal(size=4) + 1j * rng.normal(size=4), dims="component")
    assert complex(braket(a, b)) == pytest.approx(complex(np.vdot(a.values, b.values)))
    assert complex(braket(a, a)).imag == pytest.approx(0.0, abs=1e-12)
    assert complex(braket(a, a)).real == pytest.approx(float((abs(a) ** 2).sum()))


def test_braket_broadcasts_over_extra_dimensions():
    values = xr.DataArray(np.ones((3, 5, 2), dtype=complex), dims=("kx", "ky", "component"))
    result = braket(values, values)
    assert result.dims == ("kx", "ky")
    assert np.allclose(result.values, 2.0)


# --------------------------------------------------------------------------
# berry_curvature
# --------------------------------------------------------------------------


def test_berry_curvature_validates_dimension_names():
    values = xr.DataArray(np.ones((3, 3, 2), dtype=complex), dims=("kx", "ky", "component"))
    with pytest.raises(ValueError):
        berry_curvature(values, compdim="missing")
    with pytest.raises(ValueError):
        berry_curvature(values, dims=["kx", "missing"])


def test_berry_curvature_vanishes_for_a_constant_eigenvector():
    values = xr.DataArray(np.ones((6, 6, 2), dtype=complex) / np.sqrt(2),
                          dims=("kx", "ky", "component"))
    assert np.allclose(berry_curvature(values, periodic=True).values, 0.0, atol=1e-12)


def test_berry_curvature_is_gauge_invariant():
    """Multiplying each k point by its own phase must not change the curvature."""
    H = haldane_grid(t2=0.15, n=24)
    _, eigve = H.eigh()
    reference = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)

    rng = np.random.default_rng(1)
    phases = xr.DataArray(
        np.exp(2j * np.pi * rng.random(eigve.sizes["ka1"] * eigve.sizes["ka2"] * eigve.sizes["band"])
               ).reshape(eigve.sizes["ka1"], eigve.sizes["ka2"], eigve.sizes["band"]),
        dims=("ka1", "ka2", "band"),
    )
    regauged = eigve * phases
    assert np.allclose(
        berry_curvature(regauged, dims=["ka1", "ka2"], periodic=True).values,
        reference.values,
        atol=1e-9,
    )


def test_berry_curvature_of_the_two_haldane_bands_cancels():
    """The total curvature summed over all bands of a gapped model is zero."""
    H = haldane_grid(t2=0.15, n=32)
    _, eigve = H.eigh()
    berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
    assert float(berry.sum()) == pytest.approx(0.0, abs=1e-8)


def test_berry_curvature_sign_follows_the_haldane_flux():
    positive = haldane_grid(t2=0.15, n=32)
    negative = haldane_grid(t2=-0.15, n=32)
    chern = []
    for H in (positive, negative):
        _, eigve = H.eigh()
        berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
        chern.append(float(berry.sel(band=0).sum()) / 2 / np.pi)
    assert chern[0] == pytest.approx(-1, abs=1e-6)
    assert chern[1] == pytest.approx(+1, abs=1e-6)


def test_non_periodic_berry_curvature_drops_the_boundary():
    """periodic=False leaves the first row/column empty; periodic=True does not."""
    H = haldane_grid(t2=0.15, n=32)
    _, eigve = H.eigh()
    open_berry = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=False)
    assert np.allclose(open_berry.isel(ka1=0).values, 0.0)
    assert np.allclose(open_berry.isel(ka2=0).values, 0.0)

    closed = berry_curvature(eigve, dims=["ka1", "ka2"], periodic=True)
    assert not np.allclose(closed.isel(ka1=0).values, 0.0)
