import xarray as xr
from xarray.ufuncs import angle
import numpy as np


def braket(
    arr1: xr.DataArray, arr2: xr.DataArray, dim: str = "component"
) -> xr.DataArray:
    """Compute the braket of two arrays along a specified direction

    Args:
        arr1 (xr.DataArray): First array
        arr2 (xr.DataArray): Second array
        dim (str, optional): dimension of the contraction. Defaults to 'component'.

    Returns:
        xr.DataArray: The resulting array, with the dimension 'dim' contracted.
    """
    return xr.dot(arr1.conj(), arr2, dim=dim)


def two_smallest_indices(L: list[float]) -> tuple[int, int]:
    """
    Return the two smallest elements of a list.

    Parameters:
    ----------
    L: list

    Returns:
    ----------
    min1_idx, min2_idx: integers
        the indices of the two smallest elements of L
    """
    if len(L) < 2:
        return "Array must have at least two elements."

    # Get the index of the smallest element
    min1_idx = L.index(min(L))

    # Temporarily replace the smallest element to find the second smallest
    min1_value = L[min1_idx]
    L[min1_idx] = float("inf")  # Replace with infinity

    # Get the index of the second smallest element
    min2_idx = L.index(min(L))

    # Restore the original smallest element value
    L[min1_idx] = min1_value

    return min1_idx, min2_idx


def pair_nodes(NodesX: np.ndarray, NodesY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create pairs of nodes by smallest distance

    Parameters:
    ----------
    NodesX, NodesY: ndarrays, shape (n,)
        The X and Y coordinates of each node

    Returns:
    ----------
    NodesX, NodesY: ndarrays, shape (n,)
        The X and Y coordinates of each node, rearranged in pairs of increasing
        inter-distance
    """

    # Creating a distance matrix
    l = len(NodesX)
    Xs = NodesX[:, np.newaxis] - NodesX[np.newaxis, :]
    Ys = NodesY[:, np.newaxis] - NodesY[np.newaxis, :]
    Dists = Xs**2 + Ys**2

    Indxs = np.arange(l)

    indices = []

    for i in range(l // 2 - 1):
        i1, i2 = two_smallest_indices(list(Dists[0]))
        indices += [Indxs[i1], Indxs[i2]]
        Indxs = np.delete(Indxs, [i1, i2])
        Dists = np.delete(Dists, [i1, i2], 0)
        Dists = np.delete(Dists, [i1, i2], 1)

    indices += list(Indxs)
    return NodesX[indices], NodesY[indices]


def connect_indices_k(
    start: np.ndarray, stop: np.ndarray, ax: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two 1D arrays connecting start to stop, so that there is only one
    non-zero element per row/column.

    Parameters:
    ----------
    start: ndarray, shape (2,)
        Starting point coordinates
    stop: ndarray, shape (2,)
        Ending point coordinates
    ax: integer
        control wheter rows (ax = 0) of cols (ax = 1) do not contain repeated elements

    Returns:
    ----------
    rows, cols: ndarrays, shape (n,)
        Two 1D numpy arrays containing the path indices
    """

    i1, j1 = start[0], start[1]
    i2, j2 = stop[0], stop[1]

    # Calculate the number of steps needed (row/column difference + 1)
    steps = abs(i2 - i1) * (1 - ax) + abs(j2 - j1) * ax + 1

    if steps == 0:
        return np.array([i1]), np.array([i2])

    # Generate linearly spaced points between start and end
    rows = np.linspace(i1 + 0.5, i2 + 0.5, steps, dtype=int, endpoint=True)[:-1]
    cols = np.linspace(j1 + 0.5, j2 + 0.5, steps, dtype=int, endpoint=True)[:-1]

    return rows, cols


def berry_curvature(
    eigve: xr.DataArray,
    dims: list[str] = ["kx", "ky"],
    compdim: str = "component",
    periodic=False,
) -> xr.DataArray:
    """Compute the Berry curvature for all eigenvector maps stored in 'eigve', using the 4-point formula

    Args:
        eigve (xr.DataArray): A xarray DataArray containing the eigenvectors maps and some eventual additional dimensions.
        The array must at least be 3D with one dimension for the eigenvector component and two dimensions for the reciprocal space.
        dims (list, optional): The reciprocal space dimensions. Defaults to ["kx","ky"].
        compdim (str, optional): Name of the eigenvector component dimension. Default to 'component'.
        periodic (bool, optional): Whether the reciprocal grid wraps around, i.e.
        whether it tiles the Brillouin zone torus. Defaults to False.

    Note:
        The 'periodic' flag matters for any integrated quantity. With
        'periodic=False' the plaquettes touching the first row and column of the
        grid are left empty, so summing the curvature misses one edge of the
        zone and a Chern number comes out slightly below its integer value
        (unless the grid happens to duplicate its own edge, as
        'np.linspace(0, 1, n)' does). Sample the zone with
        'np.linspace(0, 1, n, endpoint=False)' and pass 'periodic=True' to get
        exact quantisation.

    Returns:
        xr.DataArray: The computed Berry curvature
    """

    if compdim not in eigve.dims:
        raise ValueError(f"{compdim} not a dimension of eigve")
    for dim in dims:
        if dim not in eigve.dims:
            raise ValueError(f"{dim} not a dimension of eigve")

    if periodic:
        eigve1 = eigve.roll({dims[0]: 1})
        eigve3 = eigve.roll({dims[1]: 1})
        eigve2 = eigve.roll({dims[0]: 1, dims[1]: 1})
    else:
        eigve1 = eigve.shift({dims[0]: 1}, fill_value=np.nan)
        eigve3 = eigve.shift({dims[1]: 1}, fill_value=np.nan)
        eigve2 = eigve.shift({dims[0]: 1, dims[1]: 1}, fill_value=np.nan)

    p1 = braket(eigve, eigve1, dim=compdim)
    p2 = braket(eigve1, eigve2, dim=compdim)
    p3 = braket(eigve2, eigve3, dim=compdim)
    p4 = braket(eigve3, eigve, dim=compdim)

    return -angle(p1 * p2 * p3 * p4).fillna(0)


def euler_curvature(
    eigve: xr.DataArray, dims: list[str] = ["kx", "ky"], compdim: str = "component"
) -> xr.DataArray:
    """
    Compute the Euler curvature of a given pair of bands

    Args:
        eigve (xr.DataArray): A xarray DataArray containing the eigenvectors maps and some eventual additional dimensions.
        The array must at least be 3D with one dimension for the eigenvector component and two dimensions for the reciprocal space.
        dims (list, optional): The reciprocal space dimensions. Defaults to ["kx","ky"].
        compdim (str, optional): Name of the eigenvector component dimension. Default to 'component'.
    Returns:
        xr.DataArray: The computed Euler curvature
    """

    # Compute the berry curvature to find the nodes, using the four-point formula
    berry = berry_curvature(eigve, dims, compdim)
    euler = (
        berry.sel(band=berry.band[:-1]) * 0
    )  # Initialize an empty euler curvature array
    euler = euler.rename({"band": "gap"})

    param_dims = [dim for dim in berry.dims if dim not in [*dims, compdim, "band"]]

    iterlist = np.meshgrid(*[berry.coords[dim] for dim in param_dims], indexing="ij")
    iterlist = [coord.reshape(-1) for coord in iterlist]
    if len(iterlist) == 0:  # check wheter there is no parameters dims
        iterlist += [[0]]
    # print(iterlist)
    for params in zip(*iterlist):
        # print(params)
        for gap in euler.gap.values:
            selection = {dim: params[i] for i, dim in enumerate(param_dims)}

            eigve_sel = eigve.sel(selection).squeeze()
            selection["band"] = gap
            berry_low = berry.sel(selection).squeeze()
            selection["band"] = gap + 1
            berry_up = berry.sel(selection).squeeze()

            # Locating the nodes by looking where pi-flux are
            Nodes_below = np.where(np.abs(berry_low) == np.pi, 0, 1)
            Nodes_above = np.where(np.abs(berry_up) == np.pi, 0, 1)

            # Creating a list of list containing the adjacent nodes below and above the principal gap
            # This will allow us to fix the gauge in such a way that the only discontinuities
            # will follow lines connecting these nodes
            Nodes_tuple = [
                list(np.where(Nodes_below + (1 - Nodes_above) == 0)),
                list(np.where(Nodes_above + (1 - Nodes_below) == 0)),
            ]

            # Define the where the Dirac strings will be

            ## Initialization
            KxLink = [np.ones_like(berry_up) for u in range(2)]
            KyLink = [np.ones_like(berry_up) for u in range(2)]
            # KxLink and KyLink store the only positions where discontinuities in the
            # gauge (along x or y) will be allowed.

            ## We define KxLink and KyLink for the lower and upper band of the gap considered
            for u in range(2):
                if len(Nodes_tuple[u][0]) % 2 == 1:
                    Nodes_tuple[u][0] = np.append(
                        Nodes_tuple[u][0], Nodes_tuple[u][0][-1]
                    )
                    Nodes_tuple[u][1] = np.append(Nodes_tuple[u][1], 0)

                if len(Nodes_tuple[u][0]) > 0:  # Check whether there is adjacent nodes
                    x_t, y_t = pair_nodes(Nodes_tuple[u][0], Nodes_tuple[u][1])

                # Loop of each pair of nodes
                for i in range(0, len(Nodes_tuple[u][0]), 2):
                    # Define the string
                    # print(x_t, y_t)
                    rowx, colx = connect_indices_k(
                        (x_t[i], y_t[i]),
                        (x_t[i + 1], y_t[i + 1]),
                        ax=0,
                    )
                    rowy, coly = connect_indices_k(
                        (x_t[i], y_t[i]), (x_t[i + 1], y_t[i + 1]), ax=1
                    )

                    KxLink[u][rowx, colx] = -1
                    KyLink[u][rowy, coly] = -1
                    # Discontinuities will be kept only where KxLink/KyLink are equal to -1

            eigve_low = eigve_sel.sel(band=gap)
            eigve_up = eigve_sel.sel(band=gap + 1)
            # Now we gauge smoothly away from the dirac strings. First, we gauge the two bands at the same time, using the strings from the lower band, then, we gauge only the upper band with its nodes
            for u in range(0, 2):
                # We convert the link arrays to xarray for easy broadcast rules
                KxLinkt = xr.DataArray(
                    KyLink[u],
                    {
                        dims[0]: eigve_sel.coords[dims[0]],
                        dims[1]: eigve_sel.coords[dims[1]],
                    },
                )
                KyLinkt = xr.DataArray(
                    np.flip(KxLink[u], axis=(0)),
                    {
                        dims[0]: eigve_sel.coords[dims[0]],
                        dims[1]: eigve_sel.coords[dims[1]],
                    },
                )

                # first we fix the gauge along the edge iky = 0

                if u == 0:  ## Only fix the lower eigenmap with its own nodes
                    jumpedge_low = eigve_low.isel(
                        {dims[1]: 0}
                    )  # jumpedge shape (n_kx, 'components')
                    jumpallowed = KxLinkt.isel({dims[1]: 0})
                    isjump = (
                        1
                        - jumpallowed
                        * xr.ufuncs.sign(
                            braket(jumpedge_low, jumpedge_low.shift({dims[0]: 1}))
                        )
                    ) / 2
                    rectify_edge = isjump.cumsum(dim=dims[0])
                    eigve_low = xr.where(
                        eigve_low.coords[dims[1]] == eigve_low.coords[dims[1]][0],
                        eigve_low * (-1) ** rectify_edge,
                        eigve_low,
                    )
                if u == 1:
                    jumpedge_up = eigve_up.isel(
                        {dims[1]: 0}
                    )  # jumpedge shape (n_kx, 'components')
                    jumpallowed = KxLinkt.isel({dims[1]: 0})
                    isjump = (
                        1
                        - jumpallowed
                        * xr.ufuncs.sign(
                            braket(jumpedge_up, jumpedge_up.shift({dims[0]: 1}))
                        )
                    ) / 2
                    rectify_edge = isjump.cumsum(dim=dims[0])
                    eigve_up = xr.where(
                        eigve_up.coords[dims[1]] == eigve_up.coords[dims[1]][0],
                        eigve_up * (-1) ** rectify_edge,
                        eigve_up,
                    )

                # Now that the edge iky = 0 is fixed, we can fix the gauge
                if u == 0:
                    isjump_low = (
                        1
                        - KyLinkt
                        * xr.ufuncs.sign(
                            braket(eigve_low, eigve_low.shift({dims[1]: 1}))
                        )
                    ) / 2
                    # isjump_low = xr.where(isjump_low.ky == isjump_low.ky[0], 0, isjump_low)
                    rectify_y_low = isjump_low.cumsum(dim=dims[1])
                    eigve_low = eigve_low * (-1) ** rectify_y_low
                if u == 1:
                    isjump_up = (
                        1
                        - KyLinkt
                        * xr.ufuncs.sign(braket(eigve_up, eigve_up.shift({dims[1]: 1})))
                    ) / 2
                    # isjump_up = xr.where(isjump_up.ky == isjump_up.ky[0], 0, isjump_up)
                    rectify_y_up = isjump_up.cumsum(dim=dims[1])
                    eigve_up = eigve_up * (-1) ** rectify_y_up

            # Now, the only discontinuities are along the Dirac strings defined by KxLink and KyLink
            # Computing the Euler curvature with the complexification method
            # print(eigve_low)
            complex_eigve = (eigve_low + 1j * eigve_up) / 2**0.5
            euler_sel = berry_curvature(complex_eigve, dims, compdim)

            final_selection = {dim: params[i] for i, dim in enumerate(param_dims)}
            final_selection["gap"] = gap
            # print(euler_sel)

            euler.loc[final_selection] = euler_sel

    return euler.fillna(0)


def euler_patch(
    eigve_low: xr.DataArray,
    eigve_up: xr.DataArray,
    dims: list[str] = ["kx", "ky"],
    compdim: str = "component",
) -> xr.DataArray:
    """Compute the patch euler invariant of the pair of eigenvector maps eigve1, eigve2. Do not support additional dimensions.

    Args:
        eigve1 (xr.DataArray): The lower band eigenvector map
        eigve2 (xr.DataArray): The upper band eigenvector map
        dims (list[str], optional): _description_. Defaults to ["kx","ky"].
        compdim (str, optional): _description_. Defaults to 'component'.

    Returns:
        xr.DataArray: An array containing the value of the patch euler invariant.
    """

    berry_low = berry_curvature(eigve_low, dims, compdim)
    berry_up = berry_curvature(eigve_up, dims, compdim)

    # Locating the nodes by looking where pi-flux are
    Nodes_below = np.where(np.abs(berry_low) == np.pi, 0, 1)
    Nodes_above = np.where(np.abs(berry_up) == np.pi, 0, 1)

    # Creating a list of list containing the adjacent nodes below and above the principal gap
    # This will allow us to fix the gauge in such a way that the only discontinuities
    # will follow lines connecting these nodes
    Nodes_list = list(np.where(Nodes_below + Nodes_above == 0))

    # Define the where the Dirac strings will be

    ## Initialization
    KxLink = np.ones_like(berry_up)
    KyLink = np.ones_like(berry_up)
    # We convert the link arrays to xarray for easy broadcast rules
    KxLink = xr.DataArray(
        KyLink, {dims[0]: eigve_low.coords[dims[0]], dims[1]: eigve_low.coords[dims[1]]}
    )
    KyLink = xr.DataArray(
        np.flip(KxLink, axis=(0)),
        {dims[0]: eigve_low.coords[dims[0]], dims[1]: eigve_low.coords[dims[1]]},
    )

    # KxLink and KyLink store the only positions where discontinuities in the
    # gauge (along x or y) will be allowed.

    ## We define KxLink and KyLink for the lower and upper band of the gap considered
    if len(Nodes_list[0]) % 2 == 1:
        raise ValueError("Odd number of nodes")

    if len(Nodes_list[0]) > 0:  # Check whether there is adjacent nodes
        x_t, y_t = pair_nodes(Nodes_list[0], Nodes_list[1])

    # Loop of each pair of nodes
    for i in range(0, len(Nodes_list[0]), 2):
        # Define the string
        # print(x_t, y_t)
        rowx, colx = connect_indices_k(
            (x_t[i], y_t[i]),
            (x_t[i + 1], y_t[i + 1]),
            ax=0,
        )
        rowy, coly = connect_indices_k((x_t[i], y_t[i]), (x_t[i + 1], y_t[i + 1]), ax=1)

        KxLink[rowx, colx] = -1
        KyLink[rowy, coly] = -1
        # Discontinuities will be kept only where KxLink/KyLink are equal to -1

    # Now we gauge smoothly away from the dirac strings.
    # first we fix the gauge along the edge iky = 0

    jumpedge_low = eigve_low.isel({dims[1]: 0})  # jumpedge shape (n_kx, 'components')
    jumpallowed = KxLink.isel({dims[1]: 0})
    isjump = (
        1
        - jumpallowed
        * xr.ufuncs.sign(braket(jumpedge_low, jumpedge_low.shift({dims[0]: 1})))
    ) / 2
    rectify_edge = isjump.cumsum(dim=dims[0])
    eigve_low = xr.where(
        eigve_low.coords[dims[1]] == eigve_low.coords[dims[1]][0],
        eigve_low * (-1) ** rectify_edge,
        eigve_low,
    )

    jumpedge_up = eigve_up.isel({dims[1]: 0})  # jumpedge shape (n_kx, 'components')
    jumpallowed = KxLink.isel({dims[1]: 0})
    isjump = (
        1
        - jumpallowed
        * xr.ufuncs.sign(braket(jumpedge_up, jumpedge_up.shift({dims[0]: 1})))
    ) / 2
    rectify_edge = isjump.cumsum(dim=dims[0])
    eigve_up = xr.where(
        eigve_up.coords[dims[1]] == eigve_up.coords[dims[1]][0],
        eigve_up * (-1) ** rectify_edge,
        eigve_up,
    )

    # Now that the edge iky = 0 is fixed, we can fix the gauge
    isjump_low = (
        1 - KyLink * xr.ufuncs.sign(braket(eigve_low, eigve_low.shift({dims[1]: 1})))
    ) / 2
    rectify_y_low = isjump_low.cumsum(dim=dims[1])

    eigve_low = eigve_low * (-1) ** rectify_y_low
    isjump_up = (
        1 - KyLink * xr.ufuncs.sign(braket(eigve_up, eigve_up.shift({dims[1]: 1})))
    ) / 2
    rectify_y_up = isjump_up.cumsum(dim=dims[1])
    eigve_up = eigve_up * (-1) ** rectify_y_up

    # Now, the only discontinuities are along the Dirac strings defined by KxLink and KyLink
    # Computing the Euler curvature with the complexification method
    # print(eigve_low)
    complex_eigve = (eigve_low + 1j * eigve_up) / 2**0.5
    euler_curv = berry_curvature(complex_eigve, dims, compdim)

    # computing the euler connection
    Ax = (
        braket(eigve_low, eigve_up.shift({dims[0]: 1}) - eigve_up.shift({dims[0]: -1}))
        / 2
    )
    Ay = (
        braket(eigve_low, eigve_up.shift({dims[1]: 1}) - eigve_up.shift({dims[1]: -1}))
        / 2
    )

    A1 = Ay.isel({dims[0]: 1, dims[1]: slice(1, -1)}).sum(dims[1])
    A2 = -(Ax.isel({dims[1]: 1, dims[0]: slice(1, -1)}).sum(dims[0]))
    A3 = -(Ay.isel({dims[0]: -2, dims[1]: slice(1, -1)}).sum(dims[1]))
    A4 = Ax.isel({dims[1]: -2, dims[0]: slice(1, -1)}).sum(dims[0])

    # performing the integrals
    integ_euler_curv = euler_curv.sum().squeeze()
    integ_euler_co = (A1 + A2 + A3 + A4).squeeze()

    return float(integ_euler_co + integ_euler_curv) / 2 / np.pi
