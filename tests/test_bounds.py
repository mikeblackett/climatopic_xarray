from typing import Literal
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import xarray as xr
import pytest as pt
import cf_xarray  # noqa: F401

from climatopic_xarray.bounds import (
    _midpoint_to_interval,
    _index_to_interval,
    _infer_interval,
    CFBoundsAccessor,
    infer_bounds,  # noqa: F401
)


@hp.given(
    start=st.integers(min_value=-10, max_value=10),
    size=st.integers(min_value=3, max_value=10),
    step=st.integers(min_value=1, max_value=3),
    closed=st.sampled_from(['left', 'right']),
)
def test_midpoint_to_interval(
    start: int,
    size: int,
    step: int,
    closed: Literal['left', 'right'],
):
    """
    Should be able to roundtrip between interval index and midpoint index.

    The values should be the same, but the types may differ (the
    inferred interval will have the dtype of the midpoint index).
    """
    # Arrange
    end = start + size * step
    expected = pd.interval_range(
        start=start, end=end, freq=step, closed=closed
    )
    index = expected.mid
    # Act
    actual = _midpoint_to_interval(index=index, closed=closed)
    # Assert
    np.testing.assert_array_equal(actual=actual, desired=expected)


class TestIndexToInterval:
    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
        label=st.sampled_from(['left', 'right']),
    )
    def test_roundtrips(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
        label: Literal['left', 'right'],
    ):
        # Arrange
        end = start + size * step
        expected = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        index = getattr(expected, label)
        # Act
        actual = _index_to_interval(index=index, closed=closed, label=label)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_delegates_to_midpoint(self):
        """
        Should delegate to _midpoint_to_interval if the label is `middle`.
        """
        # Arrange
        index = pd.Index(range(10))
        expected = _midpoint_to_interval(index=index)
        # Act
        actual = _index_to_interval(index=index, label='middle')
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)


class TestInferInterval:
    def test_infers_from_datetime_index(self):
        """
        Should infer the interval from a DatetimeIndex.
        """
        # Arrange
        index = pd.date_range(start='2000-01-01', periods=10, freq='MS')
        start = index[0]
        end = index[-1] + index.freq
        expected = pd.interval_range(
            start=start, end=end, freq=index.freqstr, closed='left'
        )
        # Act
        actual = _infer_interval(index=index)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_falls_back_to_midpoint(self):
        """
        Should fall back to _midpoint_to_interval if the datetime index has no
        frequency.
        """
        # Arrange
        index = pd.DatetimeIndex(['2000-01-01', '2000-02-01', '2000-04-01'])
        expected = _midpoint_to_interval(index=index)
        # Act
        actual = _infer_interval(index=index)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_infers_from_regular_index(self):
        """
        Should infer the interval from a regular index.
        """
        # Arrange
        start = 1
        end = 10
        closed = 'left'
        index = pd.Index(range(start, end))
        expected = pd.interval_range(
            start=start, end=end, freq=1, closed=closed
        )
        # Act
        actual = _infer_interval(index=index, closed=closed)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)


class TestInferBounds:
    def test_raises_if_not_1d(self):
        """
        Should raise an error if the dimension is not 1D.
        """

        da = xr.tutorial.open_dataset('air_temperature').air
        with pt.raises(
            ValueError,
            match='Bounds are currently only supported for 1D coordinates.',
        ):
            infer_bounds(da)

    @hp.given(
        closed=st.sampled_from(['left', 'right']),
    )
    def test_infers_bounds(self, closed: Literal['left', 'right']):
        """
        Should infer the bounds for a 1D coordinate.

        This is not heavily parameterized, as the bounds inference is
        tested in the other tests.
        """
        # Arrange
        da = xr.DataArray(
            data=[1, 2, 3], dims=['time'], coords={'time': [1, 2, 3]}
        )
        expected = xr.DataArray(
            name='time_bounds',
            data=[[1, 2], [2, 3], [3, 4]],
            dims=['time', 'bounds'],
            coords={'time': da.time.assign_attrs(bounds='time_bounds')},
            attrs={
                'closed': closed,
            },
        )
        # Act
        actual = infer_bounds(da, closed=closed)
        # Assert
        xr.testing.assert_identical(actual, expected)


class TestCFBoundsAccessor:
    def test_init(self):
        """
        Should initialize the bounds accessor.
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        xr.testing.assert_identical(ds, ds.bounds._obj)

    def test_getitem(self):
        """
        Should get the bounds for a given dimension/axis.
        """
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds('time')
        bounds = CFBoundsAccessor(ds)
        expected = ds['time_bounds']
        actual = bounds['time']
        xr.testing.assert_identical(actual, expected)

    def test_getitem_raises_for_missing_bounds(self):
        """ """
        ds = xr.tutorial.open_dataset('air_temperature')
        bounds = CFBoundsAccessor(ds)
        with pt.raises(KeyError):
            bounds['time']

    def test_axes_getter(self):
        """
        Should return a mapping of axes to bounds.
        """
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        bounds = CFBoundsAccessor(ds)
        expected = ds.cf.bounds
        actual = bounds.axes
        for k, v in actual.items():
            assert k in expected
            assert expected[k] == v

    def test_dims_getter(self):
        """
        Should return a set of names of dimensions with bounds.
        """
        dims = ['time', 'lat', 'lon']
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(dims)
        bounds = CFBoundsAccessor(ds)
        expected = set(dims)
        actual = bounds.dims
        assert actual == expected

    def test_iter(self):
        """
        Should iterate over the bounds.
        """
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        bounds = CFBoundsAccessor(ds)
        expected = sorted(ds.cf.bounds.keys())
        actual = sorted(list(iter(bounds)))
        assert actual == expected

    def test_len(self):
        # Arrange
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        bounds = CFBoundsAccessor(ds)
        expected = len(ds.cf.bounds)
        # Act
        actual = len(bounds)
        # Assert
        assert actual == expected

    def test_add_bounds_for_key(self):
        """
        Should add bounds to a DataSet.
        """
        # Arrange
        ds = xr.tutorial.open_dataset('air_temperature')
        bounds = CFBoundsAccessor(ds)
        # Act
        actual = bounds.add('time')
        # Assert
        assert 'time' in actual.bounds.dims
        assert 'T' in actual.bounds.axes

    def test_add_missing_bounds(self):
        """
        Should add all missing bounds to a DataSet.
        """
        # Arrange
        ds = xr.tutorial.open_dataset('air_temperature')
        bounds = CFBoundsAccessor(ds)
        # Act
        actual = bounds.add()
        # Assert
        assert {'time', 'lat', 'lon'} == actual.bounds.dims
        assert {'T', 'X', 'Y'} == set(actual.bounds.axes)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
    )
    def test_to_index(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
    ):
        """
        Should convert the bounds to an interval index.
        """
        # Arrange
        end = start + size * step
        expected = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        index = expected.left
        ds = xr.Dataset(
            data_vars={'foo': ('lat', range(len(index)))},
            coords={'lat': index},
        ).bounds.add('lat', closed=closed)
        # Act
        actual = ds.bounds.to_index('lat')
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
    )
    def test_to_midpoint(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
    ):
        """
        Should convert the bounds to an index of midpoints.
        """
        # Arrange
        end = start + size * step
        interval = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        expected = interval.mid
        ds = xr.Dataset(
            data_vars={'foo': ('lat', range(len(interval)))},
            coords={'lat': interval.left},
        ).bounds.add('lat', closed=closed)
        # Act
        actual = ds.bounds.to_midpoint('lat')
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)
