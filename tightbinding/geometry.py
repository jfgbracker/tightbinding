import xarray as xr
from copy import deepcopy
import numpy as np
import plotly.graph_objects as go
from types import NoneType
from typing import Union
from ipywidgets import FloatSlider, IntSlider, HBox, VBox, interactive_output
import inspect
from IPython.display import display


def make_sliders(
    data: xr.DataArray, exclude: Union[list[str], tuple[str, ...]] = (), start: str = "middle"
) -> dict:
    """Build one ipywidget slider per dimension of 'data' that is not excluded.

    Args:
        data (xr.DataArray): The array whose dimensions become sliders.
        exclude (list[str]): Dimension names that must not become sliders.
        start (str): Either 'middle' or 'left', the initial slider position.

    Returns:
        dict: A mapping of dimension name to slider widget.
    """
    sliders = {}
    for dim in [d for d in data.dims if d not in exclude]:
        coord = data.coords[dim].values
        index = 0 if start == "left" else len(coord) // 2
        if np.issubdtype(coord.dtype, np.floating):
            sliders[dim] = FloatSlider(
                min=float(coord[0]),
                max=float(coord[-1]),
                step=float((coord[-1] - coord[0]) / max(100, len(coord))),
                value=float(coord[index]),
                description=dim,
            )
        else:
            sliders[dim] = IntSlider(
                min=int(coord[0]),
                max=int(coord[-1]),
                step=1,
                value=int(coord[index]),
                description=dim,
            )
    return sliders


def stringtoarray(name: str) -> np.ndarray:
    """A simple function translating a unit cell name '['coord1' 'coord2' ...]'
    into an np.ndarray [coord1, coord2, ...]

    Args:
        name (str): the unit cell name

    Returns:
        np.ndarray: the unit cell array
    """
    array_tmp = name[1:-1].split(" ")
    array = []
    for a in array_tmp:
        if a != "":
            array += [a]
    # print(np.array(array, dtype=int))
    return np.array(array, dtype=int)


def arraytostring(array: np.ndarray) -> str:
    """The opposite as stringtoarray

    Args:
        array (np.ndarray): the unit cell array

    Returns:
        str: the unit cell name
    """

    name = "["
    for i in array:
        name += f"{i} "
    name = name[:-1] + "]"
    return name


class Orbital:
    """A class defining an orbital type"""

    def __init__(self, name: str, radius: float = 1, polarization: int = 0):
        """
        Args:
            name (str): A short name defining the orbital
            radius (float): effective radius of the orbital, used only for plotting purposes
            polarization (int): polarization state of the orbital. For plotting onl,. two orbitals on the same layers will be plotted with interferences
        """
        self.name = name
        self.radius = radius
        self.polarization = polarization
        self.angledep = (0.0,)

    def __repr__(self):
        return (
            f"Orbital {self.name}: radius = {self.radius},"
            + f" polarization = {self.polarization},"
            + f" angle dependance = {self.angledep}"
        )

    def set_angle_dependance(self, n_nodes: tuple[float]):
        """Set the angle dependance of the orbital, expressed as the number of zeros of the wavefunction between 0 and pi.
        The values given must be either integers (for cosine profile) or half integer (for sine profiles).
        This functionimplemented only for 2D lattices for now

        Args:
        n_nodes (tuple[float]): The number of nodes in the angle-dependant term of the eigenfunction.

        """
        self.angledep = n_nodes


class Site:
    """A class defining a Site object"""

    def __init__(
        self,
        name: str,
        position: Union[list[float], np.ndarray],
        orb: Union[list[Orbital], Orbital, NoneType] = None,
    ):
        """Initialize an empty Site at 'position' in the unit cell
        Args:
            name (str):
                The site name, usually same as sublattice name.
            position (np.ndarray):
                the position of the site inside its unit cell, must be given in cartesian coordinates (x,y,z,...)
            orb (Orbital or List[Orbital]):
                The orbital to be added, a list of orbitals can also be passed on, can be added later. Defaults to None
        """
        self.name = name
        self.position = np.array(position, dtype=float)
        self.orbitals: dict[str, Orbital] = dict()
        self.add_orbital([] if orb is None else orb)

    def __repr__(self):
        return (
            f"Site {self.name}: \n Position = {self.position}"
            + f"\n Orbitals: {(*[orb for orb in self.orbitals],)}"
        )

    def add_orbital(self, orb: Union[list[Orbital], Orbital]):
        """Add an orbital to the site
        Args:
            orb (Orbital or List[Orbital]): The orbital to be added, a list of orbitals can also be passed on
        """

        if isinstance(orb, list):
            for o in orb:
                if not isinstance(o, Orbital):
                    raise TypeError("Only orbital and list of orbitals are accepted")
                self.orbitals.update({o.name: o})
        elif isinstance(orb, Orbital):
            self.orbitals.update({orb.name: orb})
        else:
            raise TypeError("Only orbital and list of orbitals are accepted")
        self.define_index()

    def remove_orbital(self, name: str):
        """Remove an orbital from the site
        Args:
            name (str): The name of the orbital to be removed
        """
        self.orbitals.pop(name, None)
        self.define_index()

    def define_index(self):
        """give a unique (in the site) integer to each orbital
        inside the site
        """
        n = 0
        self.orbital_index = dict()
        for korb, orb in self.orbitals.items():
            self.orbital_index.update({korb: n})
            n += 1

    def num_orbitals(self):
        """Return the number of orbitals inside the site

        Returns:
            int: the number of orbitals inside the site
        """
        return len(self.orbitals)


class Unitcell:
    """A class defining the unit cell of a lattice"""

    def __init__(
        self,
        name: str,
        unitvecs: Union[list[list[float]], np.ndarray[float]],
        position: Union[np.ndarray[float], NoneType] = None,
    ):
        """Initialize an empty unit cell with lattice vectors unitvecs
        Args:
            name (str): The name of the unit cell
            unitvecs (np.ndarray): A (n,n)-ndarray containing the
            dimensions of the unit cell, where nthe dimensionality
            of the unit cell
            position (np.ndarray[float] or NoneType): The position of the unit cell in cartesian coordinates. default to None

        Raises:
            ValueError: If the unit cell dimension and number of unit vectors do not match
        """
        self.name = name
        self.unitvecs = np.array(unitvecs)
        self.sites: dict[Site] = dict()
        self.position = position

        s = np.shape(self.unitvecs)
        if s[0] != s[1]:
            raise ValueError("Dimension and number of unit vectors do not match")
        self.dimension = s[0]

    def __repr__(self):
        return f"{self.name} " + f"(sites {(*[site for site in self.sites],)})"

    def add_site(self, site):
        """Add a site to the unit cell
        Args:
            site (Site or List[Site]): The site to be added, a list of site can also be passed on
        """
        if isinstance(site, list):
            for s in site:
                if not isinstance(s, Site):
                    raise TypeError("Only Site and list of Sites are accepted")
                self.sites.update({s.name: deepcopy(s)})
        elif isinstance(site, Site):
            self.sites.update({site.name: deepcopy(site)})
        else:
            raise TypeError("Only Site and list of Sites are accepted")

        for ksite, site in self.sites.items():
            if len(site.position) != self.dimension:
                raise ValueError(
                    f"The position of site {ksite} does not match the dimension of the unit cell."
                )

        self.define_index()

    def remove_site(self, name: str):
        """Remove a site from the unit cell
        Args:
            name (str): The name of the site to be removed
        """
        self.sites.pop(name, None)
        self.define_index()

    def remove_orbital(self, sitename: str, orbname: str):
        """Remove a orbital from the unit cell

        Args:
            sitename (str): The name of the site in which the orbital is
            orbname (str): The name of the orbital to be removed
        """
        self.sites[sitename].remove_orbital(orbname)
        self.define_index()

    def num_sites(self):
        """Return the number of sites inside the unit cell

        Returns:
            int: the number of sites inside the unit cell
        """
        return len(self.sites)

    def num_orbitals(self):
        """Return the number of orbitals inside the unit cell

        Returns:
            int: the number of orbitals inside the unit cell
        """
        numb = 0
        for site in self.sites.values():
            numb += len(site.orbitals)
        return numb

    def reciprocal_vectors(self) -> np.ndarray:
        """Return the reciprocal lattice vectors of the unit cell.

        The returned array 'B' satisfies 'a_i . b_j = 2 pi delta_ij', i.e.
        'B = 2 pi inv(A).T' where the rows of 'A' are the unit vectors. Use
        these when converting fractional Brillouin-zone coordinates into the
        cartesian 'kx', 'ky', ... values expected by a periodic Hamiltonian:
        deriving them by hand is the most common source of wrongly sampled
        Brillouin zones.

        Returns:
            np.ndarray: A (n,n) array whose rows are the reciprocal vectors.
        """
        return 2 * np.pi * np.linalg.inv(np.asarray(self.unitvecs, dtype=float)).T

    def define_index(self):
        """Creates an index dictionary, linking each orbital of
        each site of the unit cell to an integer value.
        """
        self.orbitals_index = dict()
        i = 0
        for ksite, site in self.sites.items():
            for korb in site.orbitals:
                self.orbitals_index.update({f"{ksite}_{korb}": i})
                i += 1
        if len(self.orbitals_index) == 0:
            print(
                "/!\\ No orbitals in the unit cell, have you added orbitals to sites or sites to the unit cell?"
            )

    @staticmethod
    def combine(
        unitcells: list["Unitcell"],
        list_relpos: list[np.ndarray],
        new_unitvecs: np.ndarray,
    ) -> "Unitcell":
        """Combine a list of unit cells into a single large one
        Args:
            unitcells (list[Unitcell]): The unit cells to be added together
            list_relpos (list[np.ndarray]): the position of each unit cell relative to the first one.
            new_unitvecs (np.ndarray): The new unit vectors of the large unit cell

        Raises:
            ValueError: Dimensions of each unit cells not matching
        """
        for i in range(len(unitcells) - 1):
            if unitcells[i].dimension != unitcells[i + 1].dimension:
                raise ValueError(
                    "The dimensionalities of the unit cells must be the same"
                )
        if len(unitcells) - 1 != len(list_relpos):
            raise ValueError("Length of unitcells and list_relpos not matching")

        result_unitcell = deepcopy(unitcells[0])

        for i, (unitcell, relpos) in enumerate(zip(unitcells[1:], list_relpos)):
            for site in unitcell.sites.values():
                newsite = deepcopy(site)
                if newsite.name in result_unitcell.sites:
                    newsite.name = f"{newsite.name}-{i + 1}"
                newsite.position += relpos
                result_unitcell.add_site(newsite)

        offset = np.zeros(result_unitcell.dimension)
        for relpos in list_relpos:
            offset += relpos / len(unitcells)

        result_unitcell.unitvecs = new_unitvecs
        for site in result_unitcell.sites.values():
            site.position -= offset

        return result_unitcell

    def plot(self, visible_uc=True, display: bool = True):
        """Plots the geometry of the unit cell

        Returns:
            fig plotly.Figure: a plotly figure
        """
        if self.dimension == 1:
            raise ValueError("Do you really need me to plot a 1D unit cell?")
        elif self.dimension > 3:
            raise ValueError(
                "Too many dimensions for plotting, wait for me to get better at plotting stuff"
            )
        elif self.dimension == 2:
            Xs, Ys, Txts = [], [], []
            for ksite, site in self.sites.items():
                Xs += [site.position[0]]
                Ys += [site.position[1]]
                Txts += [f"{ksite}: {(*[korb for korb in site.orbitals],)}"]

            fig = go.Figure()

            # The unit cell outline also sets the axis ranges, so it is always
            # computed even when it is not drawn.
            ctr = (self.unitvecs[0] + self.unitvecs[1]) / 2
            ucpoly = np.array(
                [
                    np.zeros(2) - ctr,
                    self.unitvecs[0] - ctr,
                    self.unitvecs[0] + self.unitvecs[1] - ctr,
                    self.unitvecs[1] - ctr,
                    np.zeros(2) - ctr,
                ]
            )

            # Add unit cell shape
            if visible_uc:
                fig.add_trace(
                    go.Scatter(
                        x=ucpoly[:, 0],
                        y=ucpoly[:, 1],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)"),
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.2)",
                        name="Unit cell",
                        showlegend=False,
                    )
                )

            # Add sites
            fig.add_trace(
                go.Scatter(
                    x=Xs,
                    y=Ys,
                    mode="markers+text",
                    marker_color=np.arange(self.num_sites()),
                    marker_size=30,
                    text=Txts,
                    textposition="bottom center",
                    showlegend=False,
                )
            )  # hover text goes here

            fig.update_traces(textposition="bottom center")
            fig.update_layout(
                title=dict(text="Unit cell: " + self.name),
                yaxis_zeroline=False,
                xaxis_zeroline=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
                height=400,
                width=500,
                xaxis=dict(range=[np.min(ucpoly[:, 0]), np.max(ucpoly[:, 0])]),
                yaxis=dict(
                    range=[np.min(ucpoly[:, 1]), np.max(ucpoly[:, 1])], scaleanchor="x"
                ),
            )
        else:
            Xs, Ys, Zs, Txts = [], [], [], []
            for ksite, site in self.sites.items():
                Xs += [site.position[0]]
                Ys += [site.position[1]]
                Zs += [site.position[2]]
                Txts += [f"{ksite}: {(*[korb for korb in site.orbitals],)}"]

            fig = go.Figure()

            # Add unit cell shape

            index = (
                np.array(
                    [
                        [0, 1, 0, 0, 1, 0, 1, 1],
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [0, 0, 0, 1, 0, 1, 1, 1],
                    ]
                ).transpose()
                - 0.5
            )

            bounds = index @ self.unitvecs
            # print(bounds)
            # add uc
            if visible_uc:
                fig.add_trace(
                    go.Mesh3d(
                        x=bounds[:, 0],
                        y=bounds[:, 1],
                        z=bounds[:, 2],
                        alphahull=0,
                        color="black",
                        opacity=0.1,
                    )
                )

            # Add sites
            fig.add_trace(
                go.Scatter3d(
                    x=Xs,
                    y=Ys,
                    z=Zs,
                    mode="markers+text",
                    marker_color=np.arange(self.num_sites()),
                    marker_size=30,
                    text=Txts,
                    textposition="bottom center",
                )
            )  # hover text goes here

            fig.update_layout(
                title=dict(text="Unit cell: " + self.name),
                scene_aspectmode="data",
                #   yaxis_zeroline=False,
                #   xaxis_zeroline=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
        if display:
            fig.show()
        return fig


class Lattice:
    """A class that defines a simple lattice, made of a single type of unit cell"""

    def __init__(
        self,
        name: str,
        unitcell: Unitcell,
        periodicity: Union[tuple[bool, ...], NoneType] = None,
    ):
        """Initialize a simple lattice, made of a single type of unit cell
        Args:
            name (str): The name of the lattice
            unitcell (Unitcell): The unit cell of the lattice
            periodicity (tuple[bool] or None): Whether each lattice direction has
            periodic boundaries. Must have one entry per lattice dimension.
            Defaults to open boundaries in every direction.

        Raises:
            ValueError: If 'periodicity' does not have one entry per dimension
        """
        self.name = name
        self.unitcell = unitcell  # The unit cell of the lattice
        self.unitvecs = unitcell.unitvecs
        self.unitcells: dict[str, Unitcell] = (
            dict()
        )  # A dictionary that will contain all unit cells present
        self.dimension = unitcell.dimension

        if periodicity is None:
            periodicity = (False,) * self.dimension
        elif len(periodicity) != self.dimension:
            raise ValueError(
                f"'periodicity' has {len(periodicity)} entries but the unit cell "
                f"is {self.dimension}-dimensional"
            )
        self.periodicity = tuple(periodicity)
        self.mincoords = np.zeros(
            self.dimension
        )  # The smallest coordinate value in each of the dimensions. Used for periodicity checks.
        self.maxcoords = np.zeros(
            self.dimension
        )  # The largest coordinate value in each of the dimensions. Used for periodicity checks.
        self.range = np.zeros(
            self.dimension
        )  # The maximum number of repetition along each axis, /!\ do not allow to compute the number of unit cells, except if the latticeregular
        self.regular = True  # True if no site in any unit cell was displaced by end from its prescribed position. Allow for fast computation of phase factors in hoppings

    def __repr__(self):
        return f"A Lattice object with unit cell {self.unitcell.name}"

    def add_unitcell(self, ijpos: tuple, update=False):
        """Add a unit cell at a specific position in the lattice vector basis

        Args:
            ijpos (list[int]): The position of the unit cell in the lattice
            update (Bool): Whether to update the orbital index. default to False
        """

        cartpos = self.unitvecs.transpose() @ np.array(ijpos)

        name = arraytostring(ijpos)

        uc = deepcopy(self.unitcell)
        uc.position = cartpos
        self.unitcells.update({name: uc})
        self.update_bounds()
        if update:
            self.define_index()

    def create_rectangle_lattice(self, dims: tuple):
        """Generates the unitcells for a rectangular lattice with 'dims'
        repetitions
        Args:
            dims (tuple): A tuple containing the number of unit cell
            repetition along each lattice vector
        """
        grid = np.mgrid[*(slice(di) for di in dims)]
        ilist = np.array([coord.reshape(-1) for coord in grid])

        for i in range(np.shape(ilist)[1]):
            self.add_unitcell(ilist[:, i])

        self.define_index()

    def create_site_dict(self):
        """Create a dictionary containing each unit site type and linking it to a integer"""
        sitelist = []
        for kuc in self.unitcells:
            uc = self.unitcells[kuc]
            for ksite in uc.sites:
                site = uc.sites[ksite]
                if ksite not in sitelist:
                    sitelist += [ksite]

        self.site_index = dict(((sublat, i) for i, sublat in enumerate(sitelist)))

    def num_sites(self) -> int:
        """Return the number of sites in the lattice

        Returns:
            int: the number of sites in the lattice
        """
        n = 0
        for kuc in self.unitcells:
            uc = self.unitcells[kuc]
            n += uc.num_sites()
        return n

    def num_orbitals(self) -> int:
        """Return the number of orbitals in the lattice

        Returns:
            int: the number of orbitals in the lattice
        """
        n = 0
        for kuc in self.unitcells:
            uc = self.unitcells[kuc]
            n += uc.num_orbitals()
        return n

    def num_unitcells(self) -> int:
        """Return the number of unit cells in the lattice

        Returns:
            int: the number of unit cells in the lattice
        """
        return len(self.unitcells)

    def update_bounds(self):
        """Uptdate the bounds of the lattice as defined by mincoord and maxcoord"""

        Coordslist = []
        for kuc in self.unitcells:
            Coordslist += [stringtoarray(kuc)]
        Coordslist = np.array(Coordslist)
        self.mincoords = np.min(Coordslist, axis=0)
        self.maxcoords = np.max(Coordslist, axis=0)
        self.range = self.maxcoords - self.mincoords + 1

    def find_closest_uc(self, pos: np.ndarray):
        """Find the unit cell whose centerthe closest
        from the point 'pos'

        Args:
            pos (np.ndarray): a positon in cartesian coordinates

        Returns:
            _type_: the name of the closest unit cell
        """
        if not self.unitcells:
            raise ValueError("The lattice contains no unit cells")

        pos = np.asarray(pos, dtype=float)
        mindist = np.inf
        kucmin = None
        for kuc in self.unitcells:
            vec = self.unitcells[kuc].position - pos
            dist2 = vec @ vec
            if dist2 < mindist:
                mindist = dist2
                kucmin = kuc
        return kucmin

    def resolve_unit_cell(self, pos: Union[list, str], coordsys: str = "ij") -> str:
        """Return the key of the unit cell designated by 'pos'.

        Args:
            pos (list or str): The position of the unit cell, either in the
            lattice vector basis (coordsys = 'ij') or in cartesian coordinates
            (coordsys = 'xy'). If cartesian is chosen, the unit cell returned is
            the one whose center is closest to the position given. A unit cell
            key such as '[0 1]' may also be passed directly, in which case
            'coordsys' is ignored.
            coordsys (str, optional): Either 'ij' or 'xy'. Defaults to "ij".

        Returns:
            str: The name of the designated unit cell.

        Raises:
            ValueError: If 'coordsys' is neither 'ij' nor 'xy'
        """
        if isinstance(pos, str):
            return pos
        if coordsys == "ij":
            return arraytostring(np.array(pos))
        if coordsys == "xy":
            return self.find_closest_uc(pos)
        raise ValueError(f"'coordsys' must be 'ij' or 'xy', got {coordsys!r}")

    def remove_unit_cell(self, pos: list, coordsys: str = "ij"):
        """Remove a unit cell from the lattice index. 'pos' specifies
        the position of the unit cell removed, either in the lattice
        vector basis (coordsys = 'ij') or in the cartesian basis
        (coordsys = 'xy').

        Args:
            pos (list): The position of the unit cell to be removed
            coordsys (str, optional): The coordinate system of position.
            Either in lattice vector basis ('ij') or cartesian ('xy').
            If cartesian is chosen, the unit cell deleted is the one
            whose center is closest to the position given.
            Defaults to "ij".
        """
        self.unitcells.pop(self.resolve_unit_cell(pos, coordsys), None)
        self.define_index()

    def remove_site(self, pos: list, name: str, coordsys: str = "ij"):
        """Remove a site from a unit cell in the lattice. 'pos' specifies
        the position of the unit cell, either in the lattice vector basis
        (coordsys = 'ij') or in the cartesian basis
        (coordsys = 'xy').

        If the site removed was the last one in its unit cell, the unit cell
        is removed as well.

        Args:
            pos (list): The position of the unit cell.
            name (str): The name of the site to be removed.
            coordsys (str, optional): The coordinate system of position.
            Either in lattice vector basis ('ij') or cartesian ('xy').
            If cartesian is chosen, the unit cell used is the one
            whose center is closest to the position given.
            Defaults to "ij".
        """
        kuc = self.resolve_unit_cell(pos, coordsys)
        uc = self.unitcells[kuc]
        uc.sites.pop(name, None)
        uc.define_index()
        if len(uc.sites) == 0:
            # remove_unit_cell calls define_index for us
            self.remove_unit_cell(kuc)
        self.define_index()

    def remove_orbital(
        self, pos: list, sitename: str, orbname: str, coordsys: str = "ij"
    ):
        """Remove an orbital from a site of a unit cell in the lattice. 'pos'
        specifies the position of the unit cell, either in the lattice vector
        basis (coordsys = 'ij') or in the cartesian basis (coordsys = 'xy').

        If the orbital removed was the last one on its site, the site is
        removed as well.

        Args:
            pos (list): The position of the unit cell.
            sitename (str): The name of the site.
            orbname (str): The name of the orbital to be removed.
            coordsys (str, optional): The coordinate system of position.
            Either in lattice vector basis ('ij') or cartesian ('xy').
            If cartesian is chosen, the unit cell used is the one
            whose center is closest to the position given.
            Defaults to "ij".
        """
        kuc = self.resolve_unit_cell(pos, coordsys)
        uc = self.unitcells[kuc]
        uc.remove_orbital(sitename, orbname)
        if len(uc.sites[sitename].orbitals) == 0:
            self.remove_site(kuc, sitename)
        self.define_index()

    def define_index(self):
        """Creates an index dictionnay, linking each orbital of
        each site of each unit cell to an integer value.
        """
        self.orbitals_index = (
            dict()
        )  # A dictionnary that will contain all unit cells present
        n = 0
        for kuc in self.unitcells:
            uc = self.unitcells[kuc]
            for korb in uc.orbitals_index:
                self.orbitals_index.update({f"{kuc}_{korb}": n})
                n += 1
        self.reverse_orbital_index = {v: k for k, v in self.orbitals_index.items()}
        self.update_bounds()

    def evaluate_on_lattice(
        self, func, orbitals: Union[str, list[str]] = "all"
    ) -> xr.DataArray:
        """Evaluate a field on the lattice and return the evaluation as a 1D DataArray with coord 'component'.
        This functionmeant to play nice with the eigenvector DataArray generated by other classes

        Args:
            func (function):
                A function with arguments <x>, <y>,...
            orbitals (str or list[str]):
                The orbitals whose matching component will be set to func(position). Can either be 'all' to
                affect all orbitals, or a specific list or orbital names. default to 'all'

        Returns:
            xr.DataArray: the function evaluated at each lattice evaluation as a 1D DataArray with coord 'component'
        """
        sig = inspect.signature(func)

        if len(sig.parameters) != self.dimension:
            raise ValueError(
                "The number of arguments of the function do not match the dimensionality of the lattice"
            )

        # Honour an explicit return annotation, otherwise let numpy infer the
        # dtype from the values themselves (a lambda has no annotation).
        dtype = sig.return_annotation
        if dtype is inspect.Signature.empty:
            dtype = None

        Eval = np.zeros(self.num_orbitals(), dtype=object if dtype is None else dtype)
        for kuc, uc in self.unitcells.items():
            pos_uc = uc.position
            for ksite, site in uc.sites.items():
                pos = site.position + pos_uc
                if orbitals == "all":
                    for korb in site.orbitals:
                        indx = self.orbitals_index[f"{kuc}_{ksite}_{korb}"]
                        Eval[indx] = func(*pos)

                else:
                    for korb in orbitals:
                        indx = self.orbitals_index[f"{kuc}_{ksite}_{korb}"]
                        Eval[indx] = func(*pos)

        if dtype is None:
            Eval = np.asarray(Eval.tolist())

        return xr.DataArray(Eval, {"component": np.arange(self.num_orbitals())})

    def plot(self):
        """Plots the geometry of the lattice, as defined by the unitcells

        Returns:
            fig plotly.Figure: a plotly figure
        """
        if self.dimension == 1:
            raise ValueError("Do you really need me to plot a 1D lattice?")
        elif self.dimension > 3:
            raise ValueError(
                "Too many dimensions for plotting, wait for me to get better at plotting stuff"
            )
        elif self.dimension == 2:
            Xs, Ys, Cols, Txts = [], [], [], []
            self.create_site_dict()
            for kuc in self.unitcells:
                uc = self.unitcells[kuc]
                pos_uc = self.unitcells[kuc].position
                for ksite in uc.sites:
                    site = uc.sites[ksite]
                    Xs += [pos_uc[0] + site.position[0]]
                    Ys += [pos_uc[1] + site.position[1]]
                    Cols += [self.site_index[ksite]]
                    Txts += [f"{kuc}, {ksite}: {(*[korb for korb in site.orbitals],)}"]

            nsites = np.max(self.range)
            fig = go.Figure()

            # Add unit cell shape
            ctr = (self.unitcell.unitvecs[0] + self.unitcell.unitvecs[1]) / 2
            ucpoly = np.array(
                [
                    np.zeros(2) - ctr,
                    self.unitcell.unitvecs[0] - ctr,
                    self.unitcell.unitvecs[0] + self.unitcell.unitvecs[1] - ctr,
                    self.unitcell.unitvecs[1] - ctr,
                    np.zeros(2) - ctr,
                ]
            )
            if self.num_unitcells() < 501:
                for kuc in self.unitcells:
                    pos = self.unitcells[kuc].position
                    fig.add_trace(
                        go.Scatter(
                            x=ucpoly[:, 0] + pos[0],
                            y=ucpoly[:, 1] + pos[1],
                            mode="lines",
                            line=dict(color="rgba(0,0,0,1)", width=0.5),
                            name=f"Unit cell {kuc}",
                            showlegend=False,
                        )
                    )

            fig.add_trace(
                go.Scattergl(
                    x=Xs,
                    y=Ys,
                    mode="markers",
                    marker_color=Cols,
                    marker_size=max(125 / nsites, 5),
                    text=Txts,
                    showlegend=False,
                )
            )

            fig.update_traces(textposition="bottom center")
            fig.update_layout(
                title=dict(text=f"{self.name} Lattice"),
                yaxis_zeroline=False,
                xaxis_zeroline=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
                width=700,
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
        else:
            Xs, Ys, Zs, Txts, Cols = [], [], [], [], []
            self.create_site_dict()
            for kuc in self.unitcells:
                uc = self.unitcells[kuc]
                pos_uc = self.unitcells[kuc].position
                for ksite in uc.sites:
                    site = uc.sites[ksite]
                    Xs += [pos_uc[0] + site.position[0]]
                    Ys += [pos_uc[1] + site.position[1]]
                    Zs += [pos_uc[2] + site.position[2]]
                    Cols += [self.site_index[ksite]]
                    Txts += [f"{kuc}, {ksite}: {(*[korb for korb in site.orbitals],)}"]

            # Compute padded axis limits
            pad = 0.20  # 5% padding
            x_min, x_max = np.min(Xs), np.max(Xs)
            y_min, y_max = np.min(Ys), np.max(Ys)
            z_min, z_max = np.min(Zs), np.max(Zs)

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            fig = go.Figure()
            nsites = self.num_unitcells() ** 0.3
            fig.add_trace(
                go.Scatter3d(
                    x=Xs,
                    y=Ys,
                    z=Zs,
                    mode="markers",
                    marker_color=Cols,
                    marker_size=max(50 / nsites, 5),
                    text=Txts,
                    textposition="bottom center",
                )
            )

            fig.update_layout(
                title=dict(text=f"{self.name} Lattice"),
                scene_aspectmode="data",
                scene=dict(
                    xaxis=dict(range=[x_min - pad * x_range, x_max + pad * x_range]),
                    yaxis=dict(range=[y_min - pad * y_range, y_max + pad * y_range]),
                    zaxis=dict(range=[z_min - pad * z_range, z_max + pad * z_range]),
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
        return fig

    def plot_field(
        self,
        field: xr.DataArray,
        orbital: str,
        vtype: str,
        figargs=dict(),
        scatterargs=dict(),
        cbarkwargs=dict(),
    ):
        """
        Plot the value of field, a DataArray with a dimension 'component', as the color of the points in a scatter plot.
        All other dimensions are kept as sliders

        Parameters
        ----------
        field : xarray.DataArray
            Data with a dimension 'component' of the same length as the number of lattice orbitals
        orbital : string
            The name of the orbital for which to plot the value in each site
        vtype : xarray.DataArray
            The type of value to plot, determine the colormap and normalization of data
        figargs: dict
            A dictionnary passing matplotlib scatter arguments
        scatterargs: dict
            A dictionnary passing matplotlib scatter arguments
        """
        if not isinstance(field, xr.DataArray):
            raise TypeError("field must be an xarray.DataArray")

        cmap = {
            "amplitude": {"cmap": "magma", "midpoint": None},
            "amplitude_log": {"cmap": "magma", "midpoint": None},
            "real": {"cmap": "RdBu_r", "midpoint": 0},
            "imag": {"cmap": "RdBu_r", "midpoint": 0},
            "phase": {"cmap": "twilight", "midpoint": 0},
        }

        # cbarkwargs.update({"tickformat": ".2f"})

        if self.dimension == 1:
            raise ValueError("Do you really need me to plot a 1D lattice?")
        elif self.dimension > 3:
            raise ValueError(
                "Too many dimensions for plotting, wait for me to get better at plotting stuff"
            )
        elif self.dimension == 2:
            Xs, Ys, index_orbselect = [], [], []
            for kuc in self.unitcells:
                uc = self.unitcells[kuc]
                pos_uc = self.unitcells[kuc].position
                for ksite in uc.sites:
                    site = uc.sites[ksite]
                    Xs += [pos_uc[0] + site.position[0]]
                    Ys += [pos_uc[1] + site.position[1]]
                    index_orbselect += [self.orbitals_index[f"{kuc}_{ksite}_{orbital}"]]
        else:
            Xs, Ys, Zs, index_orbselect = [], [], [], []
            for kuc in self.unitcells:
                uc = self.unitcells[kuc]
                pos_uc = self.unitcells[kuc].position
                for ksite in uc.sites:
                    site = uc.sites[ksite]
                    Xs += [pos_uc[0] + site.position[0]]
                    Ys += [pos_uc[1] + site.position[1]]
                    Zs += [pos_uc[2] + site.position[2]]
                    index_orbselect += [self.orbitals_index[f"{kuc}_{ksite}_{orbital}"]]

        # point_dim = field.dims[component]
        param_dims = [dim for dim in field.dims if dim != "component"]

        # Create sliders for each parameter dimension
        sliders = make_sliders(field, exclude=["component"])

        component_sel = {"component": index_orbselect}
        # Setup figure

        initial_sel = {dim: sliders[dim].value for dim in param_dims}
        initial_sel.update(component_sel)
        colors = field.sel(initial_sel, method="nearest").values

        if self.dimension == 2:
            scatter = go.Scatter(
                x=Xs,
                y=Ys,
                mode="markers",
                marker={
                    "size": 10,
                    "color": colors,
                    "colorscale": cmap[vtype]["cmap"],
                    "cmid": cmap[vtype]["midpoint"],
                    "showscale": True,
                    "colorbar": dict(tickformat=".2f"),
                },
                # colorbar = cbarkwargs,
                hoverinfo="all",
                name="Sites",
                showlegend=False,
                **scatterargs,
            )
        else:
            scatter = go.Scatter3d(
                x=Xs,
                y=Ys,
                z=Zs,
                mode="markers",
                marker={
                    "size": 10,
                    "color": colors,
                    "colorscale": cmap[vtype]["cmap"],
                    "cmid": cmap[vtype]["midpoint"],
                    "showscale": True,
                },
                # colorbar = cbarkwargs,
                hoverinfo="all",
                name="Sites",
                showlegend=False,
                **scatterargs,
            )

        # Initialize figure
        base_fig = go.FigureWidget(
            scatter
        )  # FigureWidget is a permanent object that can be updated on the fly
        layout = dict(
            title=dict(text=f"{self.name} Lattice"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
            showlegend=False,
        )
        if self.dimension == 2:
            # The 2D axis options are meaningless for a 3D scene.
            layout.update(
                yaxis_zeroline=False,
                xaxis_zeroline=False,
                yaxis_scaleanchor="x",
            )
        else:
            layout.update(scene_aspectmode="data")
        base_fig.update_layout(**layout, **figargs)

        # Update callback
        def update(**kwargs):
            fig = base_fig
            selection = {dim: kwargs[dim] for dim in param_dims}
            selection.update(component_sel)
            selected = field.sel(selection, method="nearest").values
            with fig.batch_update():
                fig.data[0].marker["color"] = (
                    selected  # Updating the trace data[i] (link i) hoverable text
                )

        interactive_output(update, sliders)
        layout_box = VBox([base_fig] + [HBox(list(sliders.values()))])
        display(layout_box)
        return layout_box

    def split_orbital_key(self, key: str) -> tuple[str, str, str]:
        """Split a lattice orbital key into its unit cell, site and orbital parts.

        Keys have the form '<unitcell>_<site>_<orbital>'. A plain three-way
        split on '_' breaks as soon as a site or orbital name contains an
        underscore, so the site and orbital names are matched against the ones
        actually present in the unit cell.

        Args:
            key (str): An entry of 'self.orbitals_index'

        Returns:
            tuple[str, str, str]: The unit cell, site and orbital names

        Raises:
            KeyError: If the key does not name an orbital of this lattice
        """
        kuc, rest = key.split("_", 1)
        uc = self.unitcells[kuc]
        for ksite, site in uc.sites.items():
            prefix = f"{ksite}_"
            if rest.startswith(prefix) and rest[len(prefix):] in site.orbitals:
                return kuc, ksite, rest[len(prefix):]
        raise KeyError(f"{key!r} does not name an orbital of lattice {self.name!r}")

    def site_position(self, kuc: str, ksite: str) -> np.ndarray:
        """Return the absolute cartesian position of a site in the lattice."""
        return self.unitcells[kuc].position + self.unitcells[kuc].sites[ksite].position

    def _coupling_links(self, Hamiltonian: xr.DataArray) -> dict:
        """Group the non-zero off-diagonal Hamiltonian entries into site-to-site links.

        Returns a dict keyed by site pair, holding the two endpoint positions and
        the list of (orbital, orbital, matrix index) triples that contribute.
        """
        matrix = Hamiltonian
        param_dims = [dim for dim in matrix.dims if dim not in ["i", "j"]]
        if param_dims:
            # A link is drawn if it is non-zero anywhere in parameter space, so
            # that the set of traces stays fixed as the sliders move.
            occupied = (matrix != 0).any(dim=param_dims).values
        else:
            occupied = (matrix != 0).values

        rows, cols = np.nonzero(occupied)
        links = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j:
                continue
            kuc1, ksite1, korb1 = self.split_orbital_key(self.reverse_orbital_index[i])
            kuc2, ksite2, korb2 = self.split_orbital_key(self.reverse_orbital_index[j])
            key = f"{kuc1}_{ksite1}_to_{kuc2}_{ksite2}"
            if key not in links:
                links[key] = {
                    "positions": (
                        self.site_position(kuc1, ksite1),
                        self.site_position(kuc2, ksite2),
                    ),
                    "orbitals": [],
                }
            links[key]["orbitals"].append((korb1, korb2, i, j))
        return links

    @staticmethod
    def _link_text(link: dict, amplitudes: np.ndarray) -> str:
        """Build the hover label of one site-to-site link."""
        return "<br> ".join(
            f"{korb1} \u2194 {korb2} : {amplitudes[i, j].real:.2f} "
            f"+ {amplitudes[i, j].imag:.2f}i"
            for korb1, korb2, i, j in link["orbitals"]
        )

    def plot_couplings(self, Hamiltonian: xr.DataArray, width=700, height=500):
        """
        Creates an interactive Plotly figure where each link between orbitals is
        represented and can be hovered. This function extracts the information
        directly from the Hamiltonian, so it can be used for troubleshooting
        issues or forgotten couplings.

        Args:
            Hamiltonian (xr.DataArray): The Hamiltonian of the system. Each dimension
            other than "i" and "j" becomes a slider for coupling amplitude exploration.
            width (int): Figure width in pixels.
            height (int): Figure height in pixels.

        Returns:
            ipywidgets.VBox: The displayed figure and its sliders.

        Raises:
            ValueError: If the lattice is 1D or has more than 3 dimensions
            NotImplementedError: If the lattice is 3D
        """
        if self.dimension == 1:
            raise ValueError("Do you really need me to plot a 1D lattice?")
        if self.dimension > 3:
            raise ValueError(
                "Too many dimensions for plotting, wait for me to get better at plotting stuff"
            )
        if self.dimension == 3:
            raise NotImplementedError(
                "plot_couplings currently supports 2D lattices only"
            )

        self.create_site_dict()
        Xs, Ys, Txt, Cols = [], [], [], []
        # Computing the sites position and informations
        for kuc, uc in self.unitcells.items():
            pos_uc = uc.position
            for ksite, site in uc.sites.items():
                Xs += [pos_uc[0] + site.position[0]]
                Ys += [pos_uc[1] + site.position[1]]
                Txt += [f"{kuc}, {ksite}"]
                Cols += [self.site_index[ksite]]

        param_dims = [dim for dim in Hamiltonian.dims if dim not in ["i", "j"]]
        sliders = make_sliders(Hamiltonian, exclude=["i", "j"])

        # The set of links is fixed once, so trace indices stay valid as the
        # sliders move; only the hover text is refreshed on update.
        links = self._coupling_links(Hamiltonian)

        initial_sel = {dim: sliders[dim].value for dim in param_dims}
        amplitudes = Hamiltonian.sel(initial_sel, method="nearest").transpose("i", "j").values

        # Base scatter for points
        scatter = go.Scatter(
            x=Xs,
            y=Ys,
            mode="markers",
            marker=dict(size=10, color=Cols),
            hoverinfo="text",
            text=Txt,
            name="Sites",
            showlegend=False,
        )

        # Build line segments for each link
        edge_traces = []
        for link in links.values():
            pos1, pos2 = link["positions"]
            edge_traces.append(
                go.Scatter(
                    x=[pos1[0], (pos1[0] + pos2[0]) / 2, pos2[0]],
                    y=[pos1[1], (pos1[1] + pos2[1]) / 2, pos2[1]],
                    mode="lines",
                    line=dict(width=4, color="rgba(100,100,100,0.3)"),
                    name="link",
                    hoverinfo="text",
                    text=self._link_text(link, amplitudes),
                    showlegend=False,
                )
            )

        # Initial figure. FigureWidget is a permanent object that can be
        # updated on the fly.
        base_fig = go.FigureWidget(edge_traces + [scatter])
        base_fig.update_layout(
            title=dict(text=f"{self.name} Lattice"),
            xaxis=dict(constrain="domain"),
            yaxis_scaleanchor="x",
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
            width=width,
            height=height,
            showlegend=False,
        )

        # Update function, taking the interactive objects (here sliders) as
        # arguments and modifying the plot
        def update(**kwargs):
            selection = {dim: kwargs[dim] for dim in param_dims}
            selected = (
                Hamiltonian.sel(selection, method="nearest")
                .transpose("i", "j")
                .values
            )
            with base_fig.batch_update():
                for index, link in enumerate(links.values()):
                    base_fig.data[index].text = self._link_text(link, selected)

        interactive_output(update, sliders)
        layout_box = VBox([base_fig] + [HBox(list(sliders.values()))])
        display(layout_box)
        return layout_box


if __name__ == "__main__":
    a = 2.4  # the lattice constant

    # defining some orbitals
    orb_s = Orbital("s")
    orb_px = Orbital("px", 2, 0)
    orb_py = Orbital("py", 1.7, 0)
    # print(orb_px)

    # defining some sites
    ASub_2d = Site("A", np.array([-a / 2, 0]))
    ASub_2d.add_orbital([orb_s, orb_px])
    BSub_2d = Site("B", np.array([a / 2, 0]))
    BSub_2d.add_orbital(orb_s)
    CSub_2d = Site("C", np.array([0, a / 2**0.5]))
    CSub_2d.add_orbital([orb_s])
    ASub_2d.remove_orbital("px")

    # defining the honeycomb unit cell
    a1_2d = np.array([3 / 2 * a, -(3**0.5) / 2 * a])
    a2_2d = np.array([3 / 2 * a, 3**0.5 / 2 * a])
    unitvecs_2d = np.array([a1_2d, a2_2d])
    honey_uc = Unitcell("honeycomb", unitvecs_2d)
    honey_uc.add_site([ASub_2d, BSub_2d])

    new_vecs = np.array([a1_2d + a2_2d, a2_2d - a1_2d])
    # test_uc = Unitcell.combine([honey_uc, honey_uc], [a1_2d], new_vecs)
    test_uc = honey_uc
