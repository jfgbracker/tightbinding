# %%
# %matplotlib widget

import numpy as np
import xarray as xr
from typing import Union
from copy import deepcopy
from types import NoneType
from tightbinding.geometry import Lattice, stringtoarray, arraytostring


def check_valid_coord(coord) -> bool:
    """Check wheter coord is a valid coordinate array for the Hamiltonian

    Args:
        coord (_type_): the variable to check

    Returns:
        bool: True if valid
    """

    isgoodtype = isinstance(coord, xr.DataArray) or isinstance(coord, np.ndarray)
    if not isgoodtype:
        return False
    isgoodshape = len(np.shape(coord)) == 1
    return isgoodshape


class Hopping:
    """A class defining a coupling between two orbitals in
    the same lattice
    """

    def __init__(
        self,
        lattice: Lattice,
        expr,
        orbname1: str,
        orbname2: str,
        displacement: tuple[int],
    ):
        """defines a coupling term between two orbitals in the same lattice
        Args:
            orbname1 (str): The starting orbital of the hopping,
            in the unit cell naming convention "<site>_<orbital>"
            orbname2 (str): The destination orbital of the hopping,
            in the unit cell naming convention "<site>_<orbital>"
            displacement (tuple[int]): The unit cell distance between
            both orbitals
        """

        self.orb1 = orbname1
        self.orb2 = orbname2
        self.disp = displacement
        self.expr = expr
        self.lattice = lattice
        self.compute_std_distance()

    def __repr__(self):
        return (
            f"Coupling {self.orb1} to {self.orb2},"
            + f" displacement: {self.disp} unit cells"
        )

    def site_names(self) -> tuple[str, str]:
        """Return the site names of the two orbitals of this hopping.

        Orbital references have the form '<site>_<orbital>'. The site name is
        matched against the sites actually present in the unit cell so that
        names containing an underscore are handled correctly.

        Raises:
            KeyError: If either reference does not name an orbital of the unit cell
        """
        sites = self.lattice.unitcell.sites

        def resolve(reference: str) -> str:
            for ksite, site in sites.items():
                prefix = f"{ksite}_"
                if reference.startswith(prefix) and reference[len(prefix):] in site.orbitals:
                    return ksite
            raise KeyError(
                f"{reference!r} does not name an orbital of unit cell "
                f"{self.lattice.unitcell.name!r}"
            )

        return resolve(self.orb1), resolve(self.orb2)

    def compute_std_distance(self):
        """Compute the hopping distance, assuming the lattice is regular"""
        site1, site2 = self.site_names()

        dist = self.lattice.unitvecs.transpose() @ np.array(self.disp)
        dist = dist - self.lattice.unitcell.sites[site1].position
        dist = dist + self.lattice.unitcell.sites[site2].position

        self.std_dist = dist

    def number_of_jumps(
        self, start_uc: str
    ) -> tuple[np.ndarray[int], np.ndarray[float]]:
        start_uc = stringtoarray(start_uc)
        disp = np.array(self.disp)

        # Must use the same origin shift as end_unit_cell, otherwise the wrap
        # count is wrong whenever the lattice does not start at the origin.
        numbjumps = (start_uc + disp - self.lattice.mincoords) // self.lattice.range
        distance_jumped = self.lattice.unitvecs.transpose() @ (
            self.lattice.range * numbjumps
        )
        return numbjumps, distance_jumped

    def end_unit_cell(self, start_uc: str) -> tuple[str, bool, bool]:
        """determine the unit cell or the end point orbital for a
        starting orbital in the unit cell 'start_uc'

        Args:
            start_uc (str): The name of the starting unit cell

        Returns:
            end_uc (str): The name of the end point unit cell

            exists (bool): False if the end point orbital is not found in the lattice.

            allowedjump (bool): True if the hopping is not
            across a forbidden edge (open boundaries)
        """
        start_uc_array = stringtoarray(start_uc)

        end_uc_array = (
            np.mod(
                start_uc_array + np.array(self.disp) - self.lattice.mincoords,
                self.lattice.range,
            )
            + self.lattice.mincoords
        )

        # determines whether the jump is allowed
        didntjumpededge = end_uc_array == start_uc_array + np.array(self.disp)
        periodicity = np.array(self.lattice.periodicity)
        allowedjumps = np.logical_or(didntjumpededge, periodicity)
        allowedjump = bool(np.all(allowedjumps))

        end_uc = arraytostring(end_uc_array)

        # determine if the two orbitals exists
        orb1_exists = isinstance(
            self.lattice.orbitals_index.get(f"{start_uc}_{self.orb1}"), int
        )
        orb2_exists = isinstance(
            self.lattice.orbitals_index.get(f"{end_uc}_{self.orb2}"), int
        )

        return end_uc, orb1_exists * orb2_exists, allowedjump

    def compute_distance(self, start_uc: str) -> np.ndarray:
        if self.lattice.regular:
            dist = self.std_dist
            return dist
        else:
            site1, site2 = self.site_names()

            posuc1 = self.lattice.unitcells[start_uc].position
            pos1 = posuc1 + self.lattice.unitcells[start_uc].sites[site1].position

            end_uc = self.end_unit_cell(start_uc)[0]
            posuc2 = self.lattice.unitcells[end_uc].position
            pos2 = posuc2 + self.lattice.unitcells[end_uc].sites[site2].position

            distjumped = self.number_of_jumps(start_uc)[1]

            return distjumped + (pos2 - pos1)


class Energy:
    """A small class to help set the on-site energy of each orbital in the lattice"""

    def __init__(self, lattice: Lattice):
        """Initialize the on-site energy of a lattice's orbitals to 0.
        Use class functions to modify them.

        Args:
            lattice (Lattice): The lattice geometry
        """

        self.lattice = lattice
        self.onsite_energies: dict[str, str] = {
            orb: "0" for orb in lattice.orbitals_index
        }  # An index of each orbital's on site energy, in a symbolic expression form
        self.context = {"np": np}  # the context for condition evaluation

    def matching_orbitals(
        self,
        norbital: Union[str, NoneType] = None,
        nsite: Union[str, NoneType] = None,
        nunitcell: Union[str, NoneType] = None,
        condition: str = "True",
    ) -> list[str]:
        """List the orbital keys selected by a name specification.

        Any of 'norbital', 'nsite' and 'nunitcell' that is None acts as a
        wildcard, so passing none of them selects every orbital of the lattice.
        'condition' is evaluated once per site with 'position' (cartesian) and
        'index' (unit cell index) in scope.

        Args:
            norbital (str, optional): Name of the orbital. Defaults to None.
            nsite (str, optional): Name of the site. Defaults to None.
            nunitcell (str, optional): Name of the unit cell. Defaults to None.
            condition (str, optional): An additional condition on the unit cell
            index or the site position, given as an expression string.
            Defaults to "True".

        Returns:
            list[str]: The selected entries of the lattice orbital index
        """
        if nunitcell is not None:
            unitcells = {nunitcell: self.lattice.unitcells[nunitcell]}
        else:
            unitcells = self.lattice.unitcells

        selected = []
        for name_uc, uc in unitcells.items():
            index = stringtoarray(name_uc)
            if nsite is not None:
                if nsite not in uc.sites:
                    continue
                sites = {nsite: uc.sites[nsite]}
            else:
                sites = uc.sites

            for name_site, site in sites.items():
                position = uc.position + site.position
                self.context.update({"position": position, "index": index})
                if not eval(condition, {"__builtins__": {}}, self.context):
                    continue

                if norbital is not None:
                    if norbital in site.orbitals:
                        selected.append(f"{name_uc}_{name_site}_{norbital}")
                else:
                    for name_orb in site.orbitals:
                        selected.append(f"{name_uc}_{name_site}_{name_orb}")

        return selected

    def set_energy(
        self,
        expr: str,
        norbital: Union[str, NoneType] = None,
        nsite: Union[str, NoneType] = None,
        nunitcell: Union[str, NoneType] = None,
        condition: str = "True",
    ):
        """Set the on-site energy of specific orbitals of the Hamiltonian. All orbitals fitting the name
        specification (of orbital, site and unit cell) are affected. Passing no name
        specification at all affects every orbital of the lattice.

        Args:
            expr (str): The on-site energy as a function of the parameters included in the Hamiltonian.

            norbital (str, optional): Name of the orbital. Defaults to None.

            nsite (str, optional): Name of the site. Defaults to None.

            nunitcell (str, optional): Name of the unit cell. Defaults to None.

            condition (str, optional): An additional condition, either on the unit cell index or the
            site position, must be given as an expression string. Defaults to "True".
        """
        for key in self.matching_orbitals(norbital, nsite, nunitcell, condition):
            self.onsite_energies[key] = expr

    def add_energy(
        self,
        expr: str,
        norbital: Union[str, NoneType] = None,
        nsite: Union[str, NoneType] = None,
        nunitcell: Union[str, NoneType] = None,
        condition: str = "True",
    ):
        """Add the on-site energy 'expr' to the existing expression, for specific orbitals of the
        Hamiltonian. All orbitals fitting the name specification (of orbital, site and unit cell)
        are affected. Passing no name specification at all affects every orbital of the lattice.

        Args:
            expr (str): The on-site energy increment as a function of the parameters included in the Hamiltonian.

            norbital (str, optional): Name of the orbital. Defaults to None.

            nsite (str, optional): Name of the site. Defaults to None.

            nunitcell (str, optional): Name of the unit cell. Defaults to None.

            condition (str, optional): An additional condition, either on the unit cell index or the
            site position, must be given as an expression string. Defaults to "True".
        """
        for key in self.matching_orbitals(norbital, nsite, nunitcell, condition):
            # Parenthesised so that an increment such as "a if b else c" cannot
            # rebind against the existing expression.
            self.onsite_energies[key] = f"({self.onsite_energies[key]})+({expr})"


class Hamiltonianbuilder:
    """The powerhouse of this package, takes a lattice object and a list
    of Hopping terms, and construct the Hamiltonian matrix
    """

    def __init__(
        self,
        lattice: Lattice,
        params: dict[np.ndarray],
        reciprocalcoords: list[str] = [],
        real: bool = False,
    ):
        """takes a lattice object and a dict of parameters and initialize
        an empty Hamiltonian matrix

        Args:
            lattice (Lattice): the lattice describing the geometry
            params (dict[np.ndarray]): a dictionary with couples ("param", np.ndarray)
            containin the values of the parameters at which the Hamiltonian will be evaluated
            reciprocalcoords (list[str]): The list of key names for the reciprocal space
            parameter in case of periodic boundary condition. For example, if the periodicity of
            the lattice is (True,True), then two parameters must be passed (like 'kx' and 'ky')
            and must match keys in the params dict
            real (bool): True if the elements of the Hamiltonian are all real
        """
        self.periodicedges = np.sum(np.array(lattice.periodicity))
        if real:
            self.htype = np.float64
        else:
            self.htype = np.complex128

        self.parameters = deepcopy(params)
        self.recoords = reciprocalcoords
        self.lattice = lattice
        self.matdim = lattice.num_orbitals()
        self.couplings: list[Hopping] = []
        self.onsite_energies: dict[str, str] = {}
        self.parameters.update(
            {"i": np.arange(self.matdim), "j": np.arange(self.matdim)}
        )
        # name -> (dims, values), replayed by create_array on every rebuild
        self.multidim_coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self.context = {}
        self.create_array()

    def set_type(self, newtype):
        """change the type of the Hamiltonian, without rebuilding it.

        Args:
            newtype (_type_): the new type of the Hamiltonian
        """

        self.Hamiltonian = self.Hamiltonian.astype(newtype)
        self.htype = newtype

    def make_real(self):
        """Keep only the real part of the Hamiltonian, without rebuilding it."""
        self.Hamiltonian = self.Hamiltonian.real
        self.htype = float

    def create_array(self):
        """Build the (empty) xarray representing the Hamiltonian.

        Any multidimensional coordinate registered with
        :meth:`add_multidimensional_coord` is re-attached, so that rebuilding
        does not silently drop the reciprocal-space coordinates.
        """

        # "i" and "j" are always the last two dimensions, whatever order the
        # parameters were supplied or added in.
        dims = [
            (dim, coord)
            for dim, coord in self.parameters.items()
            if check_valid_coord(coord) and dim not in ("i", "j")
        ]
        dims += [(dim, self.parameters[dim]) for dim in ("i", "j")]

        self.size = tuple(len(coord) for _, coord in dims)
        self.Hamiltonian = xr.DataArray(
            np.zeros(self.size, dtype=self.htype), coords=dims
        )

        self.Hamiltonian.name = "Hamiltonian"

        for name, (coord_dims, values) in self.multidim_coords.items():
            self.Hamiltonian = self.Hamiltonian.assign_coords(
                {name: (coord_dims, values)}
            )

        self.context.update(
            {name: self.Hamiltonian.coords[name] for name, _ in dims}
        )
        self.context.update(
            {
                name: self.Hamiltonian.coords[name]
                for name in self.multidim_coords
            }
        )
        self.context.update(
            {
                name: value
                for name, value in self.parameters.items()
                if not check_valid_coord(value)
            }
        )
        self.context["np"] = np

    def update(self):
        """Update the HamiltonianBuilder in case of lattice modification, empty the Hamiltonian array"""
        self.matdim = self.lattice.num_orbitals()
        self.parameters.update(
            {"i": np.arange(self.matdim), "j": np.arange(self.matdim)}
        )
        self.create_array()

    def add_multidimensional_coord(
        self, name: str, coord: np.ndarray, dims: tuple[str]
    ):
        """Add a multidimensional coordinate to the Hamiltonian xarray. Useful for computing over non rectangular grids.

        Args:
            name (str): The name of the new coordinate
            coord (np.ndarray): The array containing the coordinate points.
            dims (tuple[str]): The dimensions on which depends this new multidimensional coordinate.
        """
        coord = np.asarray(coord)
        self.multidim_coords[name] = (tuple(dims), coord)
        self.Hamiltonian = self.Hamiltonian.assign_coords({name: (tuple(dims), coord)})

        self.context[name] = self.Hamiltonian.coords[name]

    def add_parameters(self, newparams: dict[np.ndarray]):
        """Add additional parameters to the Hamiltonian and reconstruct it empty

        Args:
            newparams (dict[np.ndarray]): The new parameters of the system
        """
        self.parameters.update(newparams)
        self.create_array()

    def add_couplings(self, couplings: Union[Hopping, list[Hopping]]):
        """Add the coupling terms of the Hamiltonian
        Args:
            couplings (Union[Hopping, list[Hopping]]): A new coupling or list of coupling
        """
        if isinstance(couplings, Hopping):
            self.couplings += [couplings]
        else:
            self.couplings += couplings

    def set_on_site_energies(self, energy: Energy):
        self.onsite_energies = deepcopy(energy.onsite_energies)

    def compute_on_site_energies(self):
        """Fill the Hamiltonian with the diagonal, on-site energy terms."""
        for key, index in self.lattice.orbitals_index.items():
            if key in self.onsite_energies:
                expr = self.onsite_energies[key]
                self.Hamiltonian.loc[{"i": index, "j": index}] += eval(
                    expr, {"__builtins__": {}}, self.context
                )

    def compute_single_link(self, coupling: Hopping, start_uc: str):
        orb1 = coupling.orb1
        orb2 = coupling.orb2
        end_uc, exist, allowed = coupling.end_unit_cell(start_uc)
        # print(start_uc, coupling.end_unit_cell(start_uc))
        if exist and allowed:
            distance = coupling.compute_distance(start_uc)
            self.context["distance"] = distance
            if self.periodicedges > 0:
                dotp = eval(
                    f"distance[0] * {self.recoords[0]}",
                    {"__builtins__": {}},
                    self.context,
                )
                for i in range(1, len(self.recoords)):
                    dotp = dotp + eval(
                        f"distance[{i}] * {self.recoords[i]}",
                        {"__builtins__": {}},
                        self.context,
                    )
                phase = np.exp(-1j * dotp)
            else:
                phase = 1

            # The angular profile of the orbitals is only defined in 2D; in any
            # other dimension the hopping is taken to be isotropic.
            angledependance = 1
            if self.lattice.dimension == 2:
                nsite1, nsite2 = coupling.site_names()
                norb1 = orb1[len(nsite1) + 1:]
                norb2 = orb2[len(nsite2) + 1:]
                theta = np.angle(distance[0] + 1j * distance[1])
                # The destination orbital lives in the end unit cell, which is
                # not the starting one whenever the hopping crosses a cell.
                dep1 = (
                    self.lattice.unitcells[start_uc]
                    .sites[nsite1]
                    .orbitals[norb1]
                    .angledep
                )
                dep2 = (
                    self.lattice.unitcells[end_uc]
                    .sites[nsite2]
                    .orbitals[norb2]
                    .angledep
                )
                ang1 = np.cos(round(dep1[0]) * theta + np.pi * (dep1[0] % 1))
                ang2 = np.cos(round(dep2[0]) * theta + np.pi * (dep2[0] % 1))
                # print(ang1, ang2)
                angledependance = ang1 * ang2
                # print(f"theta = {theta}, orbs = {orb1}, {orb2}, dep = {dep1}, {dep2}  angledep = {ang1}, {ang2}")

            strength = eval(coupling.expr, {"__builtins__": {}}, self.context)

            indx_start = self.lattice.orbitals_index[f"{start_uc}_{orb1}"]
            indx_end = self.lattice.orbitals_index[f"{end_uc}_{orb2}"]

            # print(indx_end, indx_start)

            self.Hamiltonian.loc[{"i": indx_start, "j": indx_end}] += (
                strength * phase * angledependance
            )
            self.Hamiltonian.loc[{"j": indx_start, "i": indx_end}] += np.conjugate(
                strength * phase * angledependance
            )

    def compute_couplings(self):
        for coupling in self.couplings:
            for name_uc, start_uc in self.lattice.unitcells.items():
                # print(name_uc)
                self.compute_single_link(coupling, name_uc)

    def compute_modified_links(self):
        pass

    def build(self):
        """Fill the Hamitlonian matrix using the coupling terms passed"""
        self.update()
        self.compute_on_site_energies()
        self.compute_couplings()
        self.compute_modified_links()

    def basis_exchange(self, matrix: Union[np.ndarray, list[list]]):
        """Perform a basis exchange on the Hamiltonian

        Args:
            matrix (np.ndarray or list of list): The basis exchange matrix
        """
        s = np.shape(np.array(matrix))
        if len(s) != 2 or s[0] != s[1]:
            raise ValueError("Incorrect shape for the matrix")

        V = xr.DataArray(
            np.array(matrix), coords={"i": np.arange(s[0]), "j": np.arange(s[1])}
        )
        Vctr = xr.DataArray(
            np.conjugate(np.array(matrix).T),
            coords={"i": np.arange(s[0]), "j": np.arange(s[1])},
        )
        self.Hamiltonian = xr.dot(
            Vctr.rename({"j": "k"}),
            xr.dot(
                self.Hamiltonian.rename({"i": "k", "j": "l"}),
                V.rename({"i": "l"}),
                dim=["l"],
            ),
            dim=["k"],
        )

    @staticmethod
    def gauge_factor(eigve: xr.DataArray) -> xr.DataArray:
        """Return the factor that fixes the eigenvector gauge on component 0.

        Multiplying an eigenvector map by this factor makes its first component
        real and non-negative, which removes the arbitrary phase (or sign, for a
        real Hamiltonian) returned by the solver.

        Note that this convention is degenerate wherever the first component is
        close to zero: for such lattices the resulting phase is dominated by
        numerical noise.

        Args:
            eigve (xr.DataArray): An eigenvector map with a 'component' dimension

        Returns:
            xr.DataArray: The gauge factor, broadcastable against 'eigve'
        """
        reference = eigve.isel(component=0)
        if np.iscomplexobj(eigve):
            return xr.apply_ufunc(np.exp, -1j * xr.apply_ufunc(np.angle, reference))
        return xr.apply_ufunc(np.sign, reference)

    def eigh(self):
        """A custom eigenvalue solver that redefines properly the
        names of the coordinates

        Returns:
            tuple[xr.DataArray, xr.DataArray]: The eigenvalues, with a 'band'
            dimension, and the eigenvectors, with 'band' and 'component'
            dimensions.
        """

        eigva, eigve = xr.apply_ufunc(
            np.linalg.eigh,
            self.Hamiltonian,
            input_core_dims=[["i", "j"]],
            output_core_dims=[["j"], ["i", "j"]],
        )

        eigva.name = "energy"
        eigve.name = "value"
        eigva = eigva.rename({"j": "band"})
        eigve = eigve.rename({"j": "band", "i": "component"})

        eigve = eigve * self.gauge_factor(eigve)

        return eigva, eigve

    def plot_coupling(self):
        """Plot every coupling term of the built Hamiltonian. See
        :meth:`tightbinding.geometry.Lattice.plot_couplings`."""
        return self.lattice.plot_couplings(self.Hamiltonian)

    def plot_onsite_energy(self, orbital: str):
        """Plot the on-site energies of a lattice.
        Args:
            orbital (str): The orbital to plot at each site
        """

        slicer = xr.DataArray(np.arange(self.matdim), dims="component")
        Onsites = self.Hamiltonian.isel(i=slicer, j=slicer)
        Onsites = Onsites.drop_vars(["i", "j"], errors="ignore")
        Onsites = Onsites.assign_coords(dict(component=slicer))
        return self.lattice.plot_field(Onsites.real, orbital=orbital, vtype="real")

    def eigh_parallel(self):
        """Parallelized version of the eigenvector solver, only useful for large parameter space, devours your cpu."""

        def _eigh_numpy(mat):
            vals, vecs = np.linalg.eigh(mat)
            return vals, vecs

        lendims = {dim: len(coord) for dim, coord in self.Hamiltonian.coords.items()}
        l = 0
        for dim, lencoord in lendims.items():
            if lencoord > l and dim not in ["i", "j"]:
                l = lencoord
                mxdim = dim
        chunkslen = {dim: -1 for dim in self.Hamiltonian.dims}
        chunkslen[mxdim] = 10

        Hamil = self.Hamiltonian.chunk(chunkslen, lock=True)

        # Eigenvalues of a Hermitian matrix are always real; the eigenvectors
        # follow the dtype of the Hamiltonian, so a real Hamiltonian must not
        # be promoted to complex here.
        vector_dtype = self.Hamiltonian.dtype

        eigva, eigve = xr.apply_ufunc(
            _eigh_numpy,
            Hamil,
            input_core_dims=[["i", "j"]],
            output_core_dims=[["i"], ["j", "i"]],
            dask_gufunc_kwargs={"output_sizes": {"i": lendims["i"], "j": lendims["j"]}},
            vectorize=True,  # allows broadcasting over outer dims
            dask="parallelized",  # enables dask-based parallelism
            output_dtypes=[np.float64, vector_dtype],
        )

        eigva = eigva.compute()
        eigve = eigve.compute()

        eigva.name = "energy"
        eigve.name = "value"
        eigva = eigva.rename({"i": "band"})
        eigve = eigve.rename({"i": "band", "j": "component"})

        eigve = eigve * self.gauge_factor(eigve)

        return eigva, eigve


# %%
if __name__ == "__main__":
    from tightbinding.geometry import Orbital, Site, Unitcell

    ## Define the lattice
    orb_s = Orbital("s")

    a = 2.4
    A = Site("A", np.array([-a / 2, 0]))
    A.add_orbital([orb_s])
    B = Site("B", np.array([a / 2, 0]))
    B.add_orbital(orb_s)

    a1 = np.array([3 / 2 * a, -(3**0.5) / 2 * a])
    a2 = np.array([3 / 2 * a, 3**0.5 / 2 * a])
    vecs = np.array([a1, a2])
    honey_uc = Unitcell("honeycomb", vecs)
    honey_uc.add_site([A, B])

    honeycomb = Lattice("honeycomb", honey_uc, (True, True))
    honeycomb.add_unitcell((0, 0), update=True)

    ## Construct the Hamiltonian

    # Always take the reciprocal basis from the unit cell: deriving it by hand
    # is easy to get wrong, and a wrong basis silently samples something that
    # is not the Brillouin zone.
    a1s, a2s = honey_uc.reciprocal_vectors()

    kl = 2
    ka1s = np.linspace(0, 1, 100)
    ka2s = np.linspace(0, 1, 100)
    eps0 = np.linspace(-1, 1, 51)
    t = -1

    params = {"ka1s": ka1s, "ka2s": ka2s, "eps0": eps0}

    H = Hamiltonianbuilder(honeycomb, params, ["kx", "ky"])

    Ka1s, Ka2s = np.meshgrid(ka1s, ka2s, indexing="ij")
    kx = Ka1s * a1s[0] + Ka2s * a2s[0]
    ky = Ka1s * a1s[1] + Ka2s * a2s[1]

    H.add_multidimensional_coord("kx", kx, ("ka1s", "ka2s"))
    H.add_multidimensional_coord("ky", ky, ("ka1s", "ka2s"))

    onsites = Energy(honeycomb)
    onsites.set_energy("-eps0/2", nsite="A")
    onsites.set_energy("eps0/2", nsite="B")
    H.set_on_site_energies(onsites)

    coup1 = Hopping(honeycomb, f"{t}", "A_s", "B_s", (0, 0))
    coup2 = Hopping(honeycomb, f"{t}", "A_s", "B_s", (-1, 0))
    coup3 = Hopping(honeycomb, f"{t}", "A_s", "B_s", (0, -1))

    H.add_couplings([coup1, coup2, coup3])
    H.build()

    eigva, eigve = H.eigh()
