import xarray as xr
import plotly as plt
import numpy as np
import plotly.graph_objects as go
from types import NoneType
from typing import Union
from ipywidgets import FloatSlider, IntSlider, HBox, VBox, Layout, interactive_output
from IPython.display import display
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import holoviews as hv
import hvplot.xarray

from tightbinding.geometry import make_sliders


def configure_notebook_renderer(renderer: str = "notebook_connected"):
    """Set the default plotly renderer for the current session.

    Importing this package no longer changes plotly's global configuration on
    its own, since a library should not reconfigure its host. Call this once at
    the top of a notebook if figures do not display.

    Args:
        renderer (str): A plotly renderer name. Defaults to "notebook_connected".
    """
    import plotly.io as pio

    pio.renderers.default = renderer


def plot_density(eigva: xr.DataArray, kernel_size: float = 0.1):
    """A custom function plotting the density of states of a given hamiltonian.

    Args:
        eigva (xr.DataArray): The eigenvalues of the Hamiltonian.
        kernel_size (float): the kernel size for the gaussian kernel density estimation.
    """

    slider_dims = [dim for dim in eigva.dims if dim != "band"]
    sliders = make_sliders(eigva, exclude=["band"], start="left")

    slider_selection = {dim: sliders[dim].value for dim in slider_dims}
    values = eigva.sel(slider_selection, method="nearest").values

    def compute_kde(values, n_points=200):
        """Compute KDE for the given 1D array of values."""
        values = np.asarray(values).flatten()
        kde = gaussian_kde(values, bw_method=kernel_size)
        pad = (values.max() - values.min()) * 0.10
        x_grid = np.linspace(values.min() - pad, values.max() + pad, n_points)
        y = kde.evaluate(x_grid)
        return x_grid, y

    x, y = compute_kde(values)

    base_fig = go.Figure()
    base_fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=2), fill="tozeroy", name="DoS"
        )
    )

    base_fig.update_layout(
        title="Density of States",
        xaxis_title="Energy (arbitrary units)",
        yaxis_title="Density",
        template="plotly_white",
        height=400,
    )

    density = go.FigureWidget(base_fig)

    def update(**kwargs):
        densitylocal = density
        # handle the control panel
        slider_selection = {dim: kwargs[dim] for dim in slider_dims}
        values = eigva.sel(slider_selection, method="nearest").values
        x, y = compute_kde(values)

        with densitylocal.batch_update():
            densitylocal.data[0].y = y

    interactive_output(update, sliders)

    # Display the controls + figure together
    layout_box = VBox(list(sliders.values()) + [density])
    display(layout_box)
    return layout_box


def plot_bands(bands: xr.DataArray, x: str) -> hv.DynamicMap:
    """Plot a simple cut of the band along the axis labeled 'x', and keep all others parameters as sliders

    Args:
        bands (xr.DataArray): The band structure array
        x (str): The dimension along which to plot the cut
    """
    hvplot.extension("bokeh")
    plot = bands.hvplot.line(x=x, y="energy", by="band", color="blue")
    return plot.opts(show_legend=False)


def plot_bands_3D(bands: xr.DataArray, dims=["kx", "ky"], escale=0.5):
    """Plot the full band structure as a 3D surface plot, and keep other parameters as sliders.

    Args:
        bands (xr.DataArray): The band structure array
        dims (list, optional): The dimensions along which to plot the surfaces. Defaults to ["kx", "ky"].
        escale (float, optional): a scaling factor for the energy axis. default to 0.5
    """

    slider_dims = [dim for dim in bands.dims if dim not in dims and dim != "band"]
    sliders = make_sliders(bands, exclude=[*dims, "band"], start="left")

    initial_selection = {dim: sliders[dim].value for dim in slider_dims}
    traces = []
    for b in bands.coords["band"]:
        initial_selection["band"] = b
        values = bands.sel(initial_selection).values
        cl = plt.colors.qualitative.Plotly[int(b) % 10]
        surf = go.Surface(
            x=bands.coords[dims[0]],
            y=bands.coords[dims[1]],
            z=values,
            contours={
                "z": {
                    "show": True,
                    "start": np.min(values),
                    "end": np.max(values),
                    "size": 0.05,
                }
            },
            opacity=0.9,
            colorscale=[cl, cl],
            showscale=False,
            showlegend=False,
        )
        traces += [surf]

    main_fig = go.Figure()
    main_fig.add_traces(traces)

    main_fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=escale),
            zaxis={"title": {"text": "Energy"}},
            xaxis={"title": {"text": f"{dims[0]}"}},
            yaxis={"title": {"text": f"{dims[1]}"}},
        ),
        width=800,
        height=600,
    )

    main_widget = go.FigureWidget(main_fig)

    def update(**kwargs):
        widg = main_widget
        selection = {dim: kwargs[dim] for dim in slider_dims}
        with widg.batch_update():
            for i, b in enumerate(bands.coords["band"]):
                selection["band"] = b
                values = bands.sel(selection, method="nearest").values
                widg.data[i].z = values
                widg.data[i].contours.z["start"] = np.min(values)
                widg.data[i].contours.z["end"] = np.max(values)

    interactive_output(update, sliders)

    layout_box = VBox([main_widget, HBox([slider for slider in sliders.values()])])
    display(layout_box)
    return layout_box


def plot_band_cuts(
    bands: xr.DataArray, dims=["kx", "ky"], contourindex: int = 0, res: int = 100
):
    """A custom function to plot the bands along a cut specified by two range sliders"""

    slider_dims = [dim for dim in bands.dims if dim not in dims and dim != "band"]
    sliders = make_sliders(bands, exclude=[*dims, "band"], start="left")

    rangesliders = {}
    for i, dim in enumerate(dims):
        if i:
            orien = "vertical"
        else:
            orien = "horizontal"
        coord = bands.coords[dim].values
        val1, val2 = coord[0], coord[-1]
        rangesliders[dim + "1"] = FloatSlider(
            value=val1,
            min=val1,
            max=val2,
            step=float((val2 - val1) / max(200, len(coord))),
            description=dim,
            orientation=orien,
        )
        rangesliders[dim + "2"] = FloatSlider(
            value=val2,
            min=val1,
            max=val2,
            step=float((val2 - val1) / max(200, len(coord))),
            description=dim,
            orientation=orien,
        )

    rangesliders[dims[0] + "1"].layout = Layout(width="50%", height="40px")  # bottom
    rangesliders[dims[0] + "2"].layout = Layout(width="50%", height="40px")  # bottom
    rangesliders[dims[1] + "1"].layout = Layout(width="40px", height="500px")  # side
    rangesliders[dims[1] + "2"].layout = Layout(width="40px", height="500px")  # side

    slider_selection = {dim: sliders[dim].value for dim in slider_dims}
    initial_selection = {**slider_selection, "band": contourindex}

    # create full figure
    bandcut = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=["Contour (kx, ky)", "Energy Cut"],
        horizontal_spacing=0.08,
    )

    # make the control panel
    contours = go.Contour(
        x=bands.coords[dims[0]],
        y=bands.coords[dims[1]],
        z=bands.sel(initial_selection).values,
        line=dict(color="black", dash="dash"),
        contours_coloring="none",
    )

    points = go.Scatter(
        x=[rangesliders[dims[0] + "1"].value, rangesliders[dims[0] + "2"].value],
        y=[rangesliders[dims[1] + "1"].value, rangesliders[dims[1] + "2"].value],
    )

    bandcut.add_trace(contours, col=1, row=1)
    bandcut.add_trace(points, col=1, row=1)

    # Make the cut
    kx1, kx2 = rangesliders[dims[0] + "1"].value, rangesliders[dims[0] + "2"].value
    ky1, ky2 = rangesliders[dims[1] + "1"].value, rangesliders[dims[1] + "2"].value
    kxarr = xr.DataArray(np.linspace(kx1, kx2, res), dims="z")
    kyarr = xr.DataArray(np.linspace(ky1, ky2, res), dims="z")
    cut_select = {**initial_selection}
    cut_interp = {dims[0]: kxarr, dims[1]: kyarr}
    for i in bands.coords["band"].values:
        cut_select["band"] = int(i)
        datacut = (bands.sel(cut_select)).interp(cut_interp).values.tolist()
        cut = go.Scatter(
            x=np.linspace(0, 1, res),
            y=datacut,
            line={"color": "blue"},
            mode="lines",
            showlegend=False,
        )
        bandcut.add_trace(cut, col=2, row=1)

    bandcut_widget = go.FigureWidget(bandcut)

    bandcut_widget.update_layout(
        xaxis1=dict(
            range=[
                bands.coords[dims[0]].values.min(),
                bands.coords[dims[0]].values.max(),
            ],
            scaleanchor="y1",
        ),
        yaxis1=dict(
            range=[
                bands.coords[dims[1]].values.min(),
                bands.coords[dims[1]].values.max(),
            ],
            scaleratio=1,
        ),
        yaxis_zeroline=False,
        xaxis_zeroline=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # print(slider_selection, initial_selection, cut_select, cut_interp)

    # Update callback

    def update(**kwargs):
        bandcutloc = bandcut_widget
        # handle the control panel
        slider_selection = {dim: kwargs[dim] for dim in slider_dims}
        control_selection = {**slider_selection, "band": contourindex}
        values_contour = np.transpose(
            bands.sel(control_selection, method="nearest").values
        )
        values_contour = np.squeeze(values_contour)

        # handle the main panel
        kx1, kx2 = kwargs[dims[0] + "1"], kwargs[dims[0] + "2"]
        ky1, ky2 = kwargs[dims[1] + "1"], kwargs[dims[1] + "2"]
        kxarr = xr.DataArray(np.linspace(kx1, kx2, res), dims="z")
        kyarr = xr.DataArray(np.linspace(ky1, ky2, res), dims="z")
        cut_interp = {dims[0]: kxarr, dims[1]: kyarr}
        # There is a plotly bug in C:\Users\marti\miniconda3\envs\topooptics-env\Lib\site-packages\plotly\basewidget.py, line 847, you need to comment this part for no error to be raised
        with bandcutloc.batch_update():
            bandcutloc.data[0].z = values_contour
            bandcutloc.data[1].x = [kx1, kx2]
            bandcutloc.data[1].y = [ky1, ky2]
            for i in bands.coords["band"].values:
                cut_select = {**slider_selection, "band": int(i)}
                datacut = (
                    (bands.sel(cut_select, method="nearest")).interp(cut_interp).values
                )
                bandcutloc.data[i + 2].y = datacut

    allsliders = {**sliders, **rangesliders}
    interactive_output(update, allsliders)

    layout_box = VBox(
        [allsliders[dim] for dim in slider_dims]
        + [
            HBox(
                [allsliders[dims[1] + "1"], allsliders[dims[1] + "2"], bandcut_widget]
            ),  # vertical slider + figure
            allsliders[dims[0] + "2"],
            allsliders[dims[0] + "1"],  # horizontal slider below
        ]
    )

    display(layout_box)
    return layout_box


def plot_eigenvectors(
    plots: list[list[xr.DataArray]],
    templates: list[list[Union[str, dict]]],
    titles: list[list[str]],
    eigvas: Union[list[list[Union[NoneType, xr.DataArray]]], NoneType] = None,
    dims: Union[list[str], NoneType] = None,
    excluded_dims: Union[list[str], NoneType] = None,
):
    """A general plotting function for exploring eigenvector maps. All data maps passed are rendered as a grid of plot,
    and all dimensions not in 'dims' are used as sliders controlling all plots at the same time.

    Args:
        plots (list[list[xr.DataArray]]): A list of list containing the various data to plot
        templates (list[list[Union[str,dict]]]): List of list of templates for the plotting style. Can either be a string 'phase', 'amplitude', 'symmetric',
        or a dictionary for additional control.
        titles (list[list[str]]): List of list of each subplot title.
        eigvas (list[list[Union[NoneType,xr.DataArray]]], optional): List of list of eventual band structures to overlay with the eigenvectors, can also be None for no overlay. Defaults to no overlay anywhere.
        dims (list[str], optional): The dimensions of the heatmaps. Defaults to ["kx","ky"].
        excluded_dims (list[str], optional): Dimensions that must not become sliders. Defaults to none.

    Returns:
        ipywidgets.VBox: The displayed grid of plots and its sliders.
    """
    dims = ["kx", "ky"] if dims is None else dims
    excluded_dims = [] if excluded_dims is None else excluded_dims

    n_rows = len(plots)
    n_cols = len(plots[0])

    if eigvas is None:
        eigvas = [[None] * n_cols for _ in range(n_rows)]

    if len(templates) != n_rows or len(titles) != n_rows or len(eigvas) != n_rows:
        raise ValueError(
            "'templates', 'titles' and 'eigvas' must have as many rows as 'plots'"
        )
    for i in range(n_rows):
        if len(plots[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'plots' not consistent")
        if len(templates[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'templates' not consistent")
        if len(titles[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'titles' not consistent")
        if len(eigvas[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'eigvas' not consistent")

    def create_heatmap(
        tp: xr.DataArray,
        eigva: Union[NoneType, xr.DataArray],
        template: Union[str, dict],
    ) -> tuple[dict, list[float], dict, go.Heatmap]:
        sliders = make_sliders(tp, exclude=[*dims, *excluded_dims])

        initial_sel = {dim: slider.value for dim, slider in sliders.items()}
        values = tp.sel(initial_sel, method="nearest").T

        if template == "phase":
            template = {
                "colorbar": {
                    "tickvals": [-np.pi, 0, np.pi],
                    "ticktext": ["-π", "0", "π"],
                },
                "cmap": "twilight",
                "zmin": -np.pi,
                "zmax": np.pi,
            }

        elif template == "amplitude":
            template = {
                "colorbar": {
                    "tickformat": ".0e",
                },
                "cmap": "magma",
            }

        elif template == "amplitude - saturated":
            template = {
                "colorbar": {
                    "tickformat": ".0e",
                },
                "cmap": "magma",
            }
            med = float(np.mean(abs(values)))
            values = xr.where(values > med, med, values)
            template.update({"zmax": med})

        elif template == "symmetric":
            template = {
                "colorbar": {
                    "tickformat": ".0e",
                },
                "cmap": "PiYG",
                "zmid": 0,
            }

        elif template == "symmetric - saturated":
            template = {
                "colorbar": {
                    "tickformat": ".0e",
                },
                "cmap": "PiYG",
                "zmid": 0,
            }
            med = float(np.mean(abs(values)))
            values = xr.where(values > med, med, values)
            values = xr.where(values < -med, -med, values)
            template.update({"zmin": -med, "zmax": med})

        x = tp.coords[dims[0]]
        y = tp.coords[dims[1]]

        xmin = tp.coords[dims[0]].values.min()
        xmax = tp.coords[dims[0]].values.max()
        ymin = tp.coords[dims[1]].values.min()
        ymax = tp.coords[dims[1]].values.max()

        heatmap = go.Heatmap(
            x=x,
            y=y,
            z=values,
            colorscale=template.get("cmap", "magma"),
            zmin=template.get("zmin", None),
            zmax=template.get("zmax", None),
            zmid=template.get("zmid", None),
        )
        plot = [heatmap]

        if eigva is not None:
            band_initial_sel = {
                dim: slider.value
                for dim, slider in sliders.items()
                if dim != "component" and dim in eigva.dims
            }
            band_value = eigva.sel(band_initial_sel, method="nearest").T

            contour = go.Contour(
                x=x,
                y=y,
                z=band_value,
                contours=dict(coloring="none", showlines=True),
                line=dict(color="black", width=1, dash="dash"),
                showscale=False,
            )
            plot += [contour]
        else:
            contour = go.Contour(
                x=x,
                y=y,
                z=values,
                contours=dict(coloring="none", showlines=True),
                line=dict(color="black", width=0, dash="dash"),
                showscale=False,
            )
            plot += [contour]

        return sliders, [[xmin, xmax], [ymin, ymax]], template, plot

    h_space, v_space = 0.1 + 0.1 / n_cols, 0.1 / n_rows
    main_fig = make_subplots(
        rows=n_rows, cols=n_cols, horizontal_spacing=h_space, vertical_spacing=v_space
    )

    w_pl = (1 - h_space * (n_cols - 1)) / n_cols
    h_pl = (1 - v_space * (n_rows - 1)) / n_rows

    x_p = np.linspace(w_pl / 2, 1 - w_pl / 2, n_cols)
    y_p = np.linspace(1 - h_pl / 2, h_pl / 2, n_rows)
    x_cb = [xt + w_pl / 2 for xt in x_p]
    y_t = [yt + h_pl / 2.1 for yt in y_p]

    sliders = {}
    annotations = []
    resolved_templates = {}
    for i in range(n_rows):
        for j in range(n_cols):
            tp = plots[i][j]
            eigva = eigvas[i][j]
            template = templates[i][j]
            slider, axislims, template, traces = create_heatmap(tp, eigva, template)

            cbdict = {
                **template["colorbar"],
                "len": 0.7 / n_rows,
                "x": min(1, x_cb[j]),
                "y": min(1, y_p[i]),
            }
            for trace in traces:
                if isinstance(trace, go.Heatmap):
                    trace.colorbar = cbdict

            main_fig.add_traces(traces, rows=i + 1, cols=j + 1)
            main_fig.update_xaxes(
                range=[axislims[0][0], axislims[0][1]],
                scaleanchor=f"y{i * n_cols + j + 1}",
                title=dict(text=dims[0], standoff=5),
                row=i + 1,
                col=j + 1,
            )
            main_fig.update_yaxes(
                range=[axislims[1][0], axislims[1][1]],
                scaleanchor=f"x{i * n_cols + j + 1}",
                scaleratio=1,
                title=dict(text=dims[1], standoff=5),
                row=i + 1,
                col=j + 1,
            )

            annotations.append(
                dict(
                    text=titles[i][j],
                    x=x_p[j],
                    y=min(1, y_t[i]),
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                )
            )

            sliders.update(slider)
            resolved_templates[(i, j)] = template

    main_fig.update_layout(
        yaxis_zeroline=False,
        xaxis_zeroline=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        showlegend=False,
    )
    main_fig.update_layout(
        width=400 * n_cols,
        height=350 * n_rows,
    )
    main_fig.update_layout(annotations=annotations)

    main_widget = go.FigureWidget(main_fig)

    main_widget.layout.width = 400 * n_cols
    main_widget.layout.height = 350 * n_rows

    def update_single(tp, eigva, kwargs):
        tp_dims = [
            dim for dim in tp.dims if dim not in dims and dim not in excluded_dims
        ]
        selection = {dim: kwargs[dim] for dim in tp_dims}
        values = tp.sel(selection, method="nearest").T

        band_values = values
        if eigva is not None:
            bands_dim = [
                dim
                for dim in eigva.dims
                if dim not in dims and dim not in excluded_dims
            ]
            band_selection = {dim: kwargs[dim] for dim in bands_dim}
            band_values = eigva.sel(band_selection, method="nearest").T

        return values.data, band_values.data

    def update(**kwargs):
        widg = main_widget
        values_list, bands_list = [], []
        for i in range(n_rows):
            for j in range(n_cols):
                tp = plots[i][j]
                eigva = eigvas[i][j]
                values, band_values = update_single(tp, eigva, kwargs)

                # Each subplot must be saturated with its own template.
                subplot_template = templates[i][j]
                if subplot_template == "amplitude - saturated":
                    med = float(np.mean(abs(values)))
                    values = np.where(values > med, med, values)
                elif subplot_template == "symmetric - saturated":
                    med = float(np.mean(abs(values)))
                    values = np.where(values > med, med, values)
                    values = np.where(values < -med, -med, values)

                values_list += [values]
                bands_list += [band_values]

        with widg.batch_update():
            for i, (values, band_values) in enumerate(zip(values_list, bands_list)):
                widg.data[2 * i].z = values
                widg.data[2 * i + 1].z = band_values

    interactive_output(update, sliders)
    layout_box = VBox([main_widget, HBox([slider for slider in sliders.values()])])
    display(layout_box)
    return layout_box
