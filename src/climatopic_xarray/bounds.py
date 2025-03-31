"""
climatopic_xarray.bounds
===================
Bounds module for xarray bounds accessor.

This module provides a xarray accessor for adding bounds to xarray datasets.
Bounds can be added for indexed dimension coordinates. The coordinates can be
referenced by their coordinate name or by their corresponding CF axis key.
"""
# TODO: (mike) add climatology bounds support

import warnings
from collections.abc import Iterator, Mapping
from typing import Literal

import cf_xarray  # noqa F401
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset

from climatopic_xarray.exceptions import (
    NotYetImplementedError,
)
from climatopic_xarray.frequencies import datetime_to_interval
from climatopic_xarray.typing import is_datetime_index, is_interval_closed_type
from climatopic_xarray.utilities import get_dim

TAxisKey = Literal['T', 'Z', 'Y', 'X']

CF_AXIS: set[TAxisKey] = {'T', 'Z', 'Y', 'X'}


class CFBoundsAccessor(Mapping[str, DataArray]):
    """An object for constructing bounds coordinates for xarray coordinates."""

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize a new Bounds object.

        The object is initialized every time a new dataset is created.

        Parameters
        ----------
        dataset : Dataset
            The xarray dataset to add bounds to.
        """
        self._obj: Dataset = dataset

    def __getitem__(self, item: str) -> DataArray:
        """
        Get the bounds for the specified dimension coordinate.

        The bounds are returned as a DataArray.
        The bounds can be referenced by their coordinate name or by their
        corresponding CF axis key.
        If the bounds are not found, a KeyError is raised.

        Parameters
        ----------
        item : str
            The name or CF axis key of the dimension coordinate to get bounds for.

        Returns
        -------
        DataArray
            The bounds data array.

        Raises
        ------
        KeyError
            If the bounds are not found for the specified dimension coordinate.
        """
        try:
            key = self._obj.cf.bounds[item][0]
        except KeyError as error:
            raise KeyError(f'No bounds found for axis {item!r}.') from error
        return self._obj.coords[key]

    @property
    def axes(self) -> dict[str, str]:
        """Mapping of CF axis keys to bounds-variable names."""
        return {k: v for k, v in self._obj.cf.bounds.items() if k in CF_AXIS}

    @property
    def dims(self) -> set[str]:
        """Set of dimension names for which bounds are available."""
        return {key for key in self._obj.cf.bounds if key in self._obj.dims}

    def __iter__(self) -> Iterator[str]:
        for k in self._obj.cf.bounds:
            yield str(k)

    def __len__(self):
        return len(self._obj.cf.bounds)

    def add(
        self,
        *key_args: str,
        closed: Literal['left', 'right'] | None = None,
        label: Literal['left', 'middle', 'right'] | None = None,
    ) -> Dataset:
        """
        Add bounds coordinates for the specified dimension keys.

        Returns a new object with all the original data in addition to the new
        coordinates.

        The `closed` and `label` parameters will apply to all bounds. If you
        want to add bounds with different parameters, you will need to call
        this method multiple times.

        Parameters
        ----------
        key_args : str
            The names or CF axis keys of the dimension coordinates to add
            bounds for. If no keys are provided, bounds will be added for all
            applicable dimensions.
        label : Literal['left', 'middle', 'right'], optional
            Which bin edge or midpoint the index labels.
        closed : Literal['left', 'right'], optional
            Which side of bin interval is closed.

        Returns
        -------
        Dataset
            A new dataset with bounds coordinates added.

        Raises
        ------
        KeyError
            If a key is not found in the dataset.
        """
        dataset = self._obj.copy()
        keys = set(key_args)
        if not keys:
            # default to all missing axes
            keys = set(self._obj.cf.axes) - set(self.axes)
        # normalize keys to variable names
        dims = {get_dim(obj=dataset, key=key) for key in keys}
        coords = {}
        for dim in dims:
            bounds = infer_bounds(
                dim=dim, obj=dataset[dim], closed=closed, label=label
            )
            coords.update({bounds.name: bounds})
        return dataset.assign_coords(coords)

    def to_index(self, key: str) -> pd.IntervalIndex:
        """
        Return the bounds as a pandas `IntervalIndex`.

        The closed side of the interval is determined by the `closed` attribute
        of the bounds variable. If the closed side is not specified, it defaults
        to 'left'. If the closed side attribute is not a valid
        `pandas.core.generic.IntervalClosedType`, a ValueError is raised.

        Parameters
        ----------
        key : str
            The name or CF axis key of the bounds coordinate to convert to a
            `pandas.IntervalIndex`.
        Returns
        -------
        pd.IntervalIndex
            The bounds as a `pandas.IntervalIndex`.
        """
        bounds = self[key]
        closed = bounds.attrs.get('closed', 'left')
        if is_interval_closed_type(closed):
            return pd.IntervalIndex.from_arrays(
                *bounds.values.transpose(), closed=closed
            )
        raise ValueError(
            f"Invalid closed value: {closed}. Must be 'left' or 'right'; "
            f'got {closed}'
        )

    def to_midpoint(self, key: str) -> xr.DataArray:
        """
        Return a DataArray representing the midpoints of the bounds.

        The midpoints are calculated by converting the bounds to a pandas
        `IntervalIndex`, and then taking the `IntervalIndex.mid` attribute. If
        the bounds are datetime-like, the midpoints are normalized to midnight.
        """
        interval = self.to_index(key)
        midpoint = interval.mid
        if isinstance(midpoint, pd.DatetimeIndex):
            midpoint = midpoint.normalize()  # pyright: ignore[reportAttributeAccessIssue]
        dim = get_dim(obj=self._obj, key=key)
        return xr.DataArray(
            data=midpoint,
            dims=(dim,),
            name=dim,
        )

    @property
    def _is_climatology(self) -> bool:
        if 'T' not in self._obj.cf.axes:
            return False
        return 'climatology_bounds' in self._obj.attrs


@xr.register_dataset_accessor('bounds')
class CFBoundsDatasetAccessor(CFBoundsAccessor):
    pass


@xr.register_dataarray_accessor('bounds')
class CFBoundsDataArrayAccessor:
    def __init__(self, *args, **kwargs) -> None:
        raise NotYetImplementedError(
            'bounds are not currently supported for DataArrays.'
            'To add bounds, you can convert the DataArray to a Dataset.'
        )


def infer_bounds(
    dim: str,
    obj: DataArray,
    closed: Literal['left', 'right'] | None = None,
    label: Literal['left', 'middle', 'right'] | None = None,
) -> DataArray:
    """
    Infer bounds for an indexed dimension coordinate.

    Parameters
    ----------
    dim : str
        The dimension to infer bounds for.
    obj : DataArray
        The data array to infer bounds from.
    closed : Literal['left', 'right'], optional
        The closed side of the interval.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.

    Returns
    -------
    DataArray
        The bounds data array.

    Raises
    ------
    ValueError
        If the dimension is not indexed.
    ValueError
        If the data array is not 1D.
    KeyError
        If the dimension is not found in the data array.

    """
    if obj.ndim != 1:
        raise ValueError(
            'Bounds are currently only supported for 1D coordinates.'
        )
    try:
        index = obj.get_index(dim)
    except KeyError as error:
        raise KeyError(
            'Bounds are only supported for indexed dimension coordinates.'
        ) from error

    interval = infer_interval(index=index, closed=closed, label=label)
    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    if data.shape != obj.shape + (2,):
        raise ValueError(
            f'Inferred bounds for {dim} have shape {data.shape}, '
            f'expected {obj.shape + (2,)}'
        )
    name = f'{dim}_bounds'
    coord = obj.assign_attrs(bounds=name)

    return DataArray(
        name=name,
        data=data,
        coords={dim: coord},
        dims=(dim, 'bounds'),
        attrs={'closed': interval.closed},
    )


def infer_interval(
    index: pd.Index,
    *,
    closed: Literal['left', 'right'] | None = None,
    label: Literal['left', 'middle', 'right'] | None = None,
) -> pd.IntervalIndex:
    """
    Infer an interval index from a pandas index.

    If the index is a datetime index with a regular frequency, the bounds are
    inferred from the frequency. Otherwise, the bounds are inferred assuming
    the index represents the midpoints of the intervals.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    closed : Literal['left', 'right'] | None, optional
        The closed side of the interval.
    label : Literal['left', 'middle', 'right'] | None, optional
        Which bin edge or midpoint the index labels.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    # datetime indexes will be inferred from their frequency
    if is_datetime_index(index):
        try:
            interval = datetime_to_interval(
                index=index, closed=closed, label=label
            )
        except ValueError:
            warnings.warn(
                'failed to infer bounds from datetime index frequency.'
                'falling back to midpoint inference.'
            )
            # fallback to midpoint inference for irregular datetime indexes
            return _midpoint_to_interval(index=index, closed=closed)
        else:
            return interval
    return _index_to_interval(index=index, label=label, closed=closed)


def _index_to_interval(
    index: pd.Index,
    label: Literal['left', 'middle', 'right'] | None = None,
    closed: Literal['left', 'right'] | None = None,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the bounds of an index.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    if label == 'middle':
        return _midpoint_to_interval(index=index, closed=closed)
    label = label or 'left'
    closed = closed or label
    step = np.diff(index).mean()
    if label == 'left':
        index = index.union([index[-1] + step])
    else:
        index = index.union([index[0] - step])
    return pd.IntervalIndex.from_breaks(breaks=index, closed=closed)


def _midpoint_to_interval(
    index: pd.Index | np.ndarray,
    closed: Literal['left', 'right'] | None = None,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the bounds of an index of midpoints.

    Parameters
    ----------
    index : pd.Index | np.ndarray
        The index of midpoints to infer bounds for.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    closed = closed or 'left'
    diffs = np.diff(index)
    # Assume that the first difference is the same as the second...
    diffs = np.insert(arr=diffs, obj=0, values=diffs[0])
    left = index - diffs / 2
    right = index + diffs / 2
    return pd.IntervalIndex.from_arrays(left=left, right=right, closed=closed)
