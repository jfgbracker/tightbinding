"""Small xarray helpers.

These wrap numpy ufuncs that xarray now exposes directly as ``xr.ufuncs.angle``
and ``xr.ufuncs.conjugate``; they are kept so that existing notebooks importing
them keep working.
"""

import numpy as np
import xarray as xr


def angle(data: xr.DataArray) -> xr.DataArray:
    """Element-wise argument of a complex DataArray."""
    return xr.apply_ufunc(np.angle, data)


def conjugate(data: xr.DataArray) -> xr.DataArray:
    """Element-wise complex conjugate of a DataArray."""
    return xr.apply_ufunc(np.conjugate, data)
