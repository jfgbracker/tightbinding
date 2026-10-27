"""Smoke tests for the plotting layer.

These check that figures are assembled with the right traces and data; they do
not check appearance. Every plotting entry point returns its figure or widget
container so it can be inspected here.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from conftest import GRAPHENE_NN, cubic_unitcell, graphene_bloch, honeycomb_unitcell
from tightbinding.geometry import Lattice
from tightbinding.hamiltonian import Hamiltonianbuilder, Hopping


def open_lattice(unitcell, dims):
    lat = Lattice(unitcell.name, unitcell, (False,) * unitcell.dimension)
    lat.create_rectangle_lattice(dims)
    return lat


def all_traces(container):
    """Collect plotly traces from a figure or a widget container."""
    if isinstance(container, (go.Figure, go.FigureWidget)):
        return list(container.data)
    traces = []
    for child in getattr(container, "children", []):
        traces.extend(all_traces(child))
    return traces


# --------------------------------------------------------------------------
# unit cell / lattice geometry plots
# --------------------------------------------------------------------------


@pytest.mark.parametrize("visible_uc", [True, False])
def test_unitcell_plot_2d(visible_uc):
    fig = honeycomb_unitcell().plot(visible_uc=visible_uc, display=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == (2 if visible_uc else 1)


@pytest.mark.parametrize("visible_uc", [True, False])
def test_unitcell_plot_3d(visible_uc):
    fig = cubic_unitcell().plot(visible_uc=visible_uc, display=False)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Scatter3d) for trace in fig.data)


def test_unitcell_plot_rejects_one_dimension():
    from conftest import chain_unitcell

    with pytest.raises(ValueError):
        chain_unitcell().plot(display=False)


def test_lattice_plot_2d_has_one_marker_per_site():
    lat = open_lattice(honeycomb_unitcell(), (3, 3))
    fig = lat.plot()
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert len(markers[0].x) == lat.num_sites()


def test_lattice_plot_3d_has_all_three_coordinates():
    lat = open_lattice(cubic_unitcell(), (2, 2, 2))
    fig = lat.plot()
    scatter = next(t for t in fig.data if isinstance(t, go.Scatter3d))
    assert len(scatter.x) == len(scatter.y) == len(scatter.z) == lat.num_sites()


# --------------------------------------------------------------------------
# field plots
# --------------------------------------------------------------------------


def test_plot_field_2d():
    lat = open_lattice(honeycomb_unitcell(), (3, 3))
    field = lat.evaluate_on_lattice(lambda x, y: x + y)
    container = lat.plot_field(field, orbital="s", vtype="real")
    scatter = next(t for t in all_traces(container) if isinstance(t, go.Scatter))
    assert len(scatter.x) == lat.num_sites()


def test_plot_field_3d_sets_the_z_coordinate():
    lat = open_lattice(cubic_unitcell(), (2, 2, 2))
    field = lat.evaluate_on_lattice(lambda x, y, z: x + y + z)
    container = lat.plot_field(field, orbital="s", vtype="real")
    scatter = next(t for t in all_traces(container) if isinstance(t, go.Scatter3d))
    assert scatter.z is not None, "3D field plot is missing its z coordinate"
    assert len(scatter.x) == len(scatter.y) == len(scatter.z) == lat.num_sites()


def test_plot_field_rejects_non_dataarray():
    lat = open_lattice(honeycomb_unitcell(), (2, 2))
    with pytest.raises(TypeError):
        lat.plot_field(np.zeros(8), orbital="s", vtype="real")


def test_plot_field_with_a_parameter_slider():
    import xarray as xr

    lat = open_lattice(honeycomb_unitcell(), (2, 2))
    base = lat.evaluate_on_lattice(lambda x, y: x + y)
    field = base * xr.DataArray(np.linspace(0, 1, 4), coords={"eps": np.linspace(0, 1, 4)})
    container = lat.plot_field(field, orbital="s", vtype="real")
    assert all_traces(container)


# --------------------------------------------------------------------------
# coupling plot
# --------------------------------------------------------------------------


def test_plot_couplings_draws_one_trace_per_bond():
    lat = open_lattice(honeycomb_unitcell(), (2, 2))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()
    container = H.plot_coupling()
    traces = all_traces(container)
    # 2x2 honeycomb: 4 intracell + 2 + 2 inter-cell bonds, each drawn once per
    # direction, plus one marker trace for the sites.
    assert len(traces) > 1
    assert any(t.mode == "markers" for t in traces)


def test_plot_couplings_rejects_three_dimensions():
    lat = open_lattice(cubic_unitcell(), (2, 2, 2))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", (0, 0, 0))])
    H.build()
    with pytest.raises(NotImplementedError):
        H.plot_coupling()


def test_plot_onsite_energy():
    from tightbinding.hamiltonian import Energy

    lat = open_lattice(honeycomb_unitcell(), (3, 3))
    H = Hamiltonianbuilder(lat, {"t": -1.0})
    energy = Energy(lat)
    energy.set_energy("1.0", nsite="A")
    energy.set_energy("-1.0", nsite="B")
    H.set_on_site_energies(energy)
    H.build()
    container = H.plot_onsite_energy(orbital="s")
    assert all_traces(container)


# --------------------------------------------------------------------------
# band / spectrum plots
# --------------------------------------------------------------------------


def test_plot_density():
    from tightbinding.plotting import plot_density

    lat = open_lattice(honeycomb_unitcell(), (3, 3))
    H = Hamiltonianbuilder(lat, {"t": -1.0, "eps": np.linspace(-1, 1, 4)})
    H.add_couplings([Hopping(lat, "t", "A_s", "B_s", d) for d in GRAPHENE_NN])
    H.build()
    eigva, _ = H.eigh()
    container = plot_density(eigva)
    assert all_traces(container)


def test_plot_bands():
    from tightbinding.plotting import plot_bands

    H, _ = graphene_bloch(n=8)
    eigva, _ = H.eigh()
    assert plot_bands(eigva, x="kpt") is not None


def test_plot_bands_3d_has_one_surface_per_band():
    from conftest import haldane_grid
    from tightbinding.plotting import plot_bands_3D

    H = haldane_grid(n=8)
    eigva, _ = H.eigh()
    container = plot_bands_3D(eigva, dims=["ka1", "ka2"])
    surfaces = [t for t in all_traces(container) if isinstance(t, go.Surface)]
    assert len(surfaces) == eigva.sizes["band"]


def test_plot_eigenvectors_grid():
    from conftest import haldane_grid
    from tightbinding.plotting import plot_eigenvectors

    H = haldane_grid(n=8)
    eigva, eigve = H.eigh()
    amplitude = abs(eigve.sel(band=0))
    phase = np.angle(eigve.sel(band=0))
    container = plot_eigenvectors(
        plots=[[amplitude, amplitude]],
        templates=[["amplitude", "symmetric"]],
        titles=[["a", "b"]],
        eigvas=[[eigva.sel(band=0), None]],
        dims=["ka1", "ka2"],
    )
    heatmaps = [t for t in all_traces(container) if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 2


def test_plot_eigenvectors_validates_shapes():
    from conftest import haldane_grid
    from tightbinding.plotting import plot_eigenvectors

    H = haldane_grid(n=8)
    _, eigve = H.eigh()
    amplitude = abs(eigve.sel(band=0))
    with pytest.raises(ValueError):
        plot_eigenvectors(
            plots=[[amplitude, amplitude]],
            templates=[["amplitude"]],
            titles=[["a", "b"]],
            dims=["ka1", "ka2"],
        )


def test_saturated_templates_apply_per_subplot():
    """Each subplot must use its own template, not the last one in the grid."""
    from conftest import haldane_grid
    from tightbinding.plotting import plot_eigenvectors

    H = haldane_grid(n=8)
    _, eigve = H.eigh()
    amplitude = abs(eigve.sel(band=0))
    container = plot_eigenvectors(
        plots=[[amplitude, amplitude]],
        templates=[["amplitude", "amplitude - saturated"]],
        titles=[["plain", "saturated"]],
        dims=["ka1", "ka2"],
    )
    heatmaps = [t for t in all_traces(container) if isinstance(t, go.Heatmap)]
    assert heatmaps[0].zmax is None
    assert heatmaps[1].zmax is not None
