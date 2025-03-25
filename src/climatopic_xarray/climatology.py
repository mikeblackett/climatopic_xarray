from collections.abc import Callable
from enum import StrEnum
from functools import cached_property
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from xarray import DataArray, Dataset
from xarray.core.groupby import DataArrayGroupBy, DatasetGroupBy
from xarray.core.types import QuantileMethods

from climatopic_xarray.frequencies import (
    Frequency,
    get_time_bounds,
)
from climatopic_xarray.utilities import get_dim

type FrequencyType = Literal['day', 'month', 'season']


class Method(StrEnum):
    MAX = 'max'
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    QUANTILE = 'quantile'
    STD = 'std'
    SUM = 'sum'
    VAR = 'var'


class Attrs(TypedDict):
    history: str
    climatology_bounds: tuple[str, str]
    frequency: Frequency


class Max[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Mean[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Median[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Min[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Quantile[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        q: ArrayLike,
        *,
        method: QuantileMethods = 'linear',
        keep_attrs: bool | None = None,
        skipna: bool | None = None,
        interpolation: QuantileMethods | None = None,
    ) -> T: ...


class Std[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Sum[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Var[T: (DataArray, Dataset)](Protocol):
    def __call__(
        self,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T: ...


class Climatology[T: (DataArray, Dataset)]:
    """
    An object for computing climatological statistics on xarray objects.

    This class provides methods to compute climatological statistics
    on xarray DataArray or Dataset objects. It works like the native xarray
    `groupby` object, but instead of removing the time dimension, it reduces
    it to a CF compliant climatological time axis.
    """

    def __init__(self, obj: T, *, frequency: FrequencyType) -> None:
        """Initialize a new climatology object.

        Parameters
        ----------
        obj : DataArray | Dataset
            The object to compute climatological statistics on.
        frequency : FrequencyType
            The frequency at which to compute the climatology. One of
            'day', 'month', or 'season'.
        """
        self._time_variable = get_dim(obj=obj, key='T')
        self._obj = obj.copy().cf.guess_coord_axis()
        self.frequency = Frequency(frequency).value
        self._group = f'{self._time_variable}.{self.frequency}'
        self._grouped_obj: DatasetGroupBy | DataArrayGroupBy = (
            self._obj.groupby(group=self._group)
        )

    def time(self, how: Literal['start', 'end'] = 'start') -> DataArray:
        """The time index of the climatology."""
        return self._reduce_time()

    def _reduce_time(self) -> DataArray:
        """Reduce the time dimension to a climatological time axis."""
        time = self._obj[self._time_variable]
        data = pd.to_datetime(
            [
                time[group][0].item() if group else np.nan
                for group in self._grouped_obj.groups.values()
            ]
        )
        return DataArray(
            data=data,
            dims='time',
            attrs={
                'standard_name': 'time',
                'axis': 'T',
            },
        )

    @cached_property
    def climatology_bounds(self) -> tuple[str, str]:
        """The bounds of the climatology."""
        return get_time_bounds(self._obj)

    def reduce(self, func: Callable[..., T], **kwargs) -> T:
        """
        Reduce this DataArray's data by applying a function along the time dimension.

        Parameters
        ----------
        func : Callable[..., T]
            The reduction method to apply to the grouped data.
        **kwargs : dict
            Additional keyword arguments to pass to the reduction method.

        Returns
        -------
        T
            The reduced xarray object.
        """
        result = (
            func(dim='time', **kwargs)
            .swap_dims({self.frequency: 'time'})
            .assign_coords(time=self.time())
            .sortby(variables='time')
        )
        return result

    @overload
    def __getattr__(self, name: Literal['max']) -> Max[T]: ...

    @overload
    def __getattr__(self, name: Literal['mean']) -> Mean[T]: ...

    @overload
    def __getattr__(self, name: Literal['median']) -> Median[T]: ...

    @overload
    def __getattr__(self, name: Literal['min']) -> Min[T]: ...

    @overload
    def __getattr__(self, name: Literal['quantile']) -> Quantile[T]: ...

    @overload
    def __getattr__(self, name: Literal['std']) -> Std[T]: ...

    @overload
    def __getattr__(self, name: Literal['sum']) -> Sum[T]: ...

    @overload
    def __getattr__(self, name: Literal['var']) -> Var[T]: ...

    def __getattr__(self, name: str) -> Any:
        if name in Method:
            method = Method(name)

            def bound_func(**kwargs) -> T:
                return self.reduce(
                    func=self._get_reducer(method=method), **kwargs
                )

            return bound_func
        # Delegate to the native groupby object
        try:
            attr = getattr(self._grouped_obj, name)
            if not isinstance(attr, Callable):
                return attr
        except AttributeError as error:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from error

    def _get_reducer(self, method: Method) -> Callable[..., T]:
        """
        Get the reduction method to apply to the grouped data.

        Returns
        -------
        Callable[..., T]
            The reduction method to apply to the grouped data.
        """
        # This is needed to support WeightedClimatology...
        return getattr(self._grouped_obj, method)

    def __repr__(self) -> str:
        name = type(self).__name__
        labels = "', '".join(
            [str(object=label) for label in self._grouped_obj.groups]
        )
        return f"<{name}, grouped over {self.frequency}s, with labels: '{labels}'>"


class WeightedClimatology[T: (DataArray, Dataset)](Climatology[T]):
    def __init__(
        self,
        obj: T,
        *,
        frequency: FrequencyType,
    ) -> None:
        # Assign the weights as a coordinate so they are grouped with the data
        self._weights = _calculate_time_weights(
            obj=obj.time, frequency=frequency
        )
        obj = obj.assign_coords({'time_weights': self._weights})
        super().__init__(
            obj=obj,
            frequency=frequency,
        )

    @property
    def weights(self) -> DataArray:
        """The time-weights used to compute the weighted climatology.

        The weights represent the fraction of the time period that each
        time point contributes to the climatological average."""
        return self._weights

    def _get_reducer(self, method: Method) -> Callable[..., T]:
        # TODO (mike): This is a bit hacky! It will be replaced once
        #  xarray provides a way to compose weighted groupby operations
        #  (See: https://github.com/pydata/xarray/issues/3937).
        #  For now, we use `groupby.map` and the native `weighted` methods
        #  to reduce each group. It's not the most efficient implementation,
        #  but it avoids having to reimplement all the weighted reduction
        #  methods.
        def reducer(**kwargs) -> T:
            result = self._grouped_obj.map(
                lambda obj: getattr(
                    obj.weighted(weights=obj.time_weights), method
                )(**kwargs)
            )
            return cast(T, result)

        return reducer


class DataArrayClimatology(Climatology[DataArray]):
    """An object for computing climatological statistics on DataArray
    objects."""

    pass


class DataArrayWeightedClimatology(WeightedClimatology[DataArray]):
    """An object for computing weighted climatological statistics on DataArray
    objects."""

    pass


class DatasetClimatology(Climatology[Dataset]):
    """An object for computing climatological statistics on Dataset
    objects."""

    pass


class DatasetWeightedClimatology(WeightedClimatology[Dataset]):
    """An object for computing weighted climatological statistics on Dataset
    objects."""

    pass


@xr.register_dataarray_accessor('climatology')
class DataArrayClimatologyAccessor:
    def __init__(self, obj: DataArray) -> None:
        self._obj: DataArray = obj

    @overload
    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: Literal[True],
    ) -> DataArrayWeightedClimatology: ...

    @overload
    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: Literal[False],
    ) -> DataArrayClimatology: ...

    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: bool = False,
    ) -> DataArrayWeightedClimatology | DataArrayClimatology:
        if weighted:
            return DataArrayWeightedClimatology(
                obj=self._obj,
                frequency=frequency,
            )
        return DataArrayClimatology(obj=self._obj, frequency=frequency)


@xr.register_dataset_accessor('climatology')
class DatasetClimatologyAccessor:
    def __init__(self, obj: Dataset) -> None:
        self._obj: Dataset = obj

    @overload
    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: Literal[True],
    ) -> DatasetWeightedClimatology: ...

    @overload
    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: Literal[False],
    ) -> DatasetClimatology: ...

    def __call__(
        self,
        frequency: FrequencyType,
        *,
        weighted: bool = True,
    ) -> DatasetClimatology | DatasetWeightedClimatology:
        if weighted:
            return DatasetWeightedClimatology(
                obj=self._obj,
                frequency=frequency,
            )
        return DatasetClimatology(obj=self._obj, frequency=frequency)


def _calculate_time_weights(
    obj: DataArray, frequency: FrequencyType
) -> DataArray:
    """Return an array of time weights for the given frequency.

    The weights represent the fraction of the time period that each time point
    """
    try:
        bounds = obj.coords[obj.time.attrs['bounds']]
    except KeyError:
        bounds = obj._to_temp_dataset().bounds.infer_bounds('T')
    deltas = bounds.diff(dim='bounds').squeeze().astype(dtype=np.float64)
    group = f'time.{frequency}'
    deltas_grouped = deltas.groupby(group=group)
    weights = (
        (deltas_grouped / deltas_grouped.sum())
        .drop_attrs()
        .rename('time_weights')
    )
    # Validate the weights: the weights for each group should sum to 1
    sum_of_weights = weights.groupby(group=group).sum()
    expected_sum_of_weights = np.ones(shape=len(sum_of_weights))
    np.testing.assert_allclose(
        actual=sum_of_weights, desired=expected_sum_of_weights
    )
    return weights.drop_vars('time')
