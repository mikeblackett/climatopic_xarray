from collections.abc import Callable
from typing import Literal
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr

import hypothesis as hp
from hypothesis import strategies as st

from climatopic_xarray.frequencies import (
    datetime_to_interval,
    get_freq,
    get_offset,
    get_period_alias,
    get_time_bounds,
    infer_freq,
    infer_midpoint_freq,
    to_datetime_index,
)
from climatopic_xarray.testing.strategies._offset_aliases import (
    OffsetCategories,
)
from climatopic_xarray.testing.strategies.frequencies import (
    datetime_indexes,
    offset_aliases,
)


class TestToDatetimeIndex:
    """
    Tests for to_datetime_index function.
    """

    @pt.mark.parametrize('klass', [xr.DataArray, np.array])
    def test_raises_if_not_1_dimensional(self, klass: Callable):
        """
        Should raise a ValueError if the index is not 1D.
        """
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        data = klass(arrays)
        with pt.raises(ValueError):
            to_datetime_index(data)

    @pt.mark.parametrize('klass', [xr.DataArray, pd.Series, np.array])
    def test_raises_if_index_not_datetime_like(self, klass: Callable):
        """
        Should raise a ValueError if the index is not a datetime-like.
        """
        index = pd.Index(range(10))
        data = klass(index)

        with pt.raises(ValueError):
            to_datetime_index(data)

    @pt.mark.parametrize('klass', [xr.DataArray, pd.Series, np.array])
    @hp.given(index=datetime_indexes(min_size=1))
    def test_returns_datetime_index(
        self, index: pd.DatetimeIndex, klass: Callable
    ):
        """
        Should return a datetime index.
        """
        data = klass(index)
        result = to_datetime_index(data)
        assert isinstance(result, pd.DatetimeIndex)
        np.testing.assert_array_equal(actual=result, desired=index)


class TestInferMidpointFreq:
    """
    Tests for infer_midpoint_freq function.
    """

    def test_raises_if_index_is_not_datetime(self):
        """
        Should raise a ValueError if the index is not a datetime-like.
        """
        index = pd.Index([1, 2, 3])
        with pt.raises(ValueError):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(index=datetime_indexes(max_size=3))
    def test_raises_if_index_is_too_short(self, index: pd.DatetimeIndex):
        """
        Should raise a ValueError if the index is too short.
        """
        with pt.raises(ValueError):
            infer_midpoint_freq(index)

    @hp.given(data=st.data())
    def test_correctly_infers_freq(self, data: st.DataObject):
        """
        Should correctly infer the frequency of a datetime index of midpoints
        """
        how: Literal['start', 'end'] = data.draw(
            st.sampled_from(['start', 'end'])
        )
        exclude_categories: OffsetCategories = (
            ['ME', 'QE', 'YE'] if how == 'start' else ['MS', 'QS', 'YS']
        )
        freqs = offset_aliases(
            min_n=1, max_n=3, exclude_categories=exclude_categories
        )
        index = data.draw(
            datetime_indexes(
                freqs=freqs,
                min_size=5,
                normalize=True,
            )
        )
        interval = pd.IntervalIndex.from_breaks(index)
        midpoint = pd.to_datetime(interval.mid)

        inferred_freq = infer_midpoint_freq(index=midpoint, how=how)

        # It should be able to infer a frequency string
        assert isinstance(inferred_freq, str)
        try:
            # For most frequencies, the inferred frequency should be equal to
            # the frequency of the original index.
            assert inferred_freq == index.freqstr
        except AssertionError:
            # However, some frequencies are equivalent, but not equal.
            # For example, 3MS would be inferred as QS.
            # In this case we can test equivalence by recreating the
            # original index using the inferred frequency.
            actual = pd.date_range(
                start=index[0],
                periods=index.size,
                freq=inferred_freq,
            )
            np.testing.assert_array_equal(actual, index)


class TestInferFreq:
    """
    Tests for infer_freq function.
    """

    def test_raises_if_index_not_datetime_like(self):
        """
        Should raise a TypeError if the index is not a datetime-like.
        """
        index = pd.Index(range(10))
        with pt.raises(TypeError):
            infer_freq(index)  # type: ignore

    def test_returns_none_for_sub_daily_frequency(self):
        """
        Should return None for sub-daily frequencies.
        """
        index = pd.date_range(start='2000', periods=10, freq='h')
        assert infer_freq(index) is None

    @hp.given(
        index=datetime_indexes(min_size=3),
    )
    def test_infers_freq(
        self,
        index: pd.DatetimeIndex,
    ):
        """
        Should infer the frequency of a regular datetime index.

        The inferred freq can be equivalent, but not equal, to the original
        index freq. We can test equivalence by recreating the original index
        using the inferred freq and comparing the two.
        """
        freq = infer_freq(index)
        assert isinstance(freq, str)
        actual = pd.DatetimeIndex(data=index.values, freq=freq)
        np.testing.assert_array_equal(actual=actual, desired=index)

    @hp.given(data=st.data())
    def test_infers_freq_from_midpoint(
        self,
        data: st.DataObject,
    ):
        """
        Should infer the frequency of a regular datetime index of midpoints.
        """
        type_ = data.draw(st.sampled_from(['index', 'series', 'data_array']))
        freqs = offset_aliases(
            min_n=1, max_n=3, exclude_categories=['ME', 'QE', 'YE']
        )
        index = data.draw(
            datetime_indexes(
                freqs=freqs,
                min_size=5,
                normalize=True,
            )
        )
        interval = pd.IntervalIndex.from_breaks(index)
        mid = pd.to_datetime(interval.mid)
        # If the index has an inferrable freq, it wouldn't take the
        # `infer_midpoint_freq` path...
        hp.assume(pd.infer_freq(mid) is None)

        if type_ == 'index':
            midpoint = mid
        elif type_ == 'series':
            midpoint = mid.to_series()
        else:
            midpoint = mid.to_series().to_xarray()

        freq = infer_freq(midpoint)
        assert isinstance(freq, str)
        actual = pd.DatetimeIndex(data=index.values, freq=freq)
        np.testing.assert_array_equal(actual, index)


class TestGetFreq:
    """
    Tests for get_freq function.
    """

    def test_raises_if_index_not_datetime_like(self):
        """
        Should raise a TypeError if the index is not a datetime-like.
        """
        index = pd.Index(range(10))
        with pt.raises(TypeError):
            get_freq(index)  # type: ignore

    def test_raises_if_no_inferred_frequency(self):
        """
        Should raise a ValueError if the index has no inferred frequency.
        """
        index = pd.DatetimeIndex(
            ['2000-01-02', '2000-01-03', '2000-01-01', '2000-01-04']
        )
        with pt.raises(ValueError):
            get_freq(index)

    @hp.given(index=datetime_indexes(min_size=4))
    def test_returns_explicit_freq(self, index: pd.DatetimeIndex):
        """
        Should return the frequency of a regular datetime index directly

        The inferred freq of a datetime index can be different (yet equivalent)
        from the index's explicit freq. If the index has an explicit freq, we
        should return it directly, instead of inferring it.
        """
        hp.assume(index.freq != pd.infer_freq(index))
        assert get_freq(index) == index.freqstr

    @hp.given(
        index=datetime_indexes(min_size=4),
    )
    def test_infers_freq(
        self,
        index: pd.DatetimeIndex,
    ):
        """
        Should infer the frequency of a regular datetime index.

        The inferred freq can be equivalent, but not equal to the original
        index freq. We can test equivalence by recreating the original index
        using the inferred freq and comparing the two.
        """
        freq = get_freq(index)
        actual = pd.DatetimeIndex(data=index, freq=freq)
        np.testing.assert_array_equal(actual=actual, desired=index)


class TestGetOffset:
    """Tests for get_offset function."""

    @pt.mark.parametrize(
        'freq',
        [
            'abc',
            '123',
            'MS-JAN',
        ],
    )
    def test_raises_given_invalid_freq(self, freq: str):
        """
        Should raise a ValueError given an unsupported offset alias.
        """
        with pt.raises(ValueError):
            get_offset(freq)  # type: ignore

    @hp.given(freq=offset_aliases())
    def test_converts_valid_freqs(self, freq: str):
        """
        Should convert valid frequencies to their period alias string.
        """
        assert isinstance(get_offset(freq), pd.DateOffset)


class TestGetPeriodAlias:
    """Tests for get_period_alias function."""

    @pt.mark.parametrize(
        'freq',
        [
            'abc',
            '123',
            'MS-JAN',
            'ms',
            '-2MS',
        ],
    )
    def test_raises_given_invalid_freq(self, freq: str):
        """
        Should raise a ValueError given an unsupported offset alias.
        """
        with pt.raises(ValueError):
            get_period_alias(freq)

    @hp.given(freqs=offset_aliases())
    def test_converts_valid_freqs(self, freqs: str):
        """
        Should convert valid frequencies to their period alias string.
        """
        assert isinstance(get_period_alias(freqs), str)

    @pt.mark.parametrize(
        'freq, expected',
        [
            ('D', 'D'),
            ('W-MON', 'W-SUN'),
            ('W-WED', 'W-TUE'),
            ('MS', 'M'),
            ('ME', 'M'),
            ('YS', 'Y-DEC'),
            ('YE', 'Y-DEC'),
            ('YS-JAN', 'Y-DEC'),
            ('YE-DEC', 'Y-DEC'),
            ('QS-DEC', 'Q-NOV'),
        ],
    )
    def test_returns_correct_period_alias(self, freq: str, expected: str):
        """
        Should return the correct period alias for the given offset alias.
        """
        assert get_period_alias(freq) == expected

    @pt.mark.parametrize(
        'freq, expected',
        [
            ('QS', 'Q-DEC'),
            ('QE', 'Q-DEC'),
            ('YS', 'Y-DEC'),
            ('YE', 'Y-DEC'),
        ],
    )
    def test_returns_default_anchors(self, freq: str, expected: str):
        """
        Should return period aliases with their default anchor.
        """
        assert get_period_alias(freq) == expected

    @pt.mark.parametrize('quantifier', list(range(2, 100, 13)))
    @pt.mark.parametrize(
        'freq, expected',
        [
            ('D', 'D'),
            ('W-MON', 'W-SUN'),
            ('W-WED', 'W-TUE'),
            ('MS', 'M'),
            ('ME', 'M'),
            ('YS', 'Y-DEC'),
            ('YE', 'Y-DEC'),
            ('YS-JAN', 'Y-DEC'),
            ('YE-DEC', 'Y-DEC'),
            ('QS-DEC', 'Q-NOV'),
        ],
    )
    def test_handles_quantifiers(
        self, freq: str, expected: str, quantifier: int
    ):
        """
        Should handle quantifiers in the offset alias.
        """
        assert (
            get_period_alias(f'{quantifier}{freq}')
            == f'{quantifier}{expected}'
        )


class TestDatetimeToInterval:
    """Tests for datetime_to_interval function."""

    def test_raises_if_no_inferred_frequency(self):
        """
        Should raise a ValueError if the index has no inferred frequency.
        """
        index = pd.DatetimeIndex(['2000-01-02', '2000-01-03', '2000-01-01'])
        with pt.raises(ValueError):
            datetime_to_interval(index)

    @hp.given(data=st.data())
    def test_should_not_raise_with_regular_frequency(
        self, data: st.DataObject
    ):
        """
        Should infer intervals from a datetime index with a regular frequency.
        """
        index = data.draw(datetime_indexes(min_size=4))
        label: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )
        closed: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )
        datetime_to_interval(
            index=index,
            label=label,
            closed=closed,
        )

    @hp.given(data=st.data())
    def test_should_not_raise_with_midpoint_frequency(
        self, data: st.DataObject
    ):
        """
        Should infer intervals from a datetime index with a midpoint frequency.
        """
        index = data.draw(datetime_indexes(min_size=5))
        closed: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )
        midpoint = pd.to_datetime(pd.IntervalIndex.from_breaks(index).mid)
        datetime_to_interval(
            index=midpoint,
            label='middle',
            closed=closed,
        )

    @hp.given(data=st.data())
    def test_intervals_are_contiguous(self, data: st.DataObject):
        """
        Should create intervals that are contiguous.
        """
        index = data.draw(datetime_indexes(min_size=3))
        label: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )
        closed: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )

        result = datetime_to_interval(
            index=index,
            label=label,
            closed=closed,
        )
        pd.testing.assert_index_equal(
            left=result.left[1:], right=result.right[:-1]
        )

    @hp.given(data=st.data())
    def test_intervals_contain_index_elements(self, data: st.DataObject):
        """
        The intervals should contain the original index elements
        if label == closed.

        If label != closed, the intervals will not contain the elements
        """
        side: Literal['left', 'right'] | None = data.draw(
            st.sampled_from(['left', 'right', None])
        )
        if side == 'left':
            freqs = offset_aliases(categories=('D', 'MS', 'QS', 'YS'))
        else:
            freqs = offset_aliases(categories=('W', 'ME', 'QE', 'YE'))

        index = data.draw(datetime_indexes(freqs=freqs, min_size=3))

        try:
            result = datetime_to_interval(
                index=index,
                label=side,
                closed=side,
            )
        except OverflowError:
            pt.skip('out of bounds datetime false negative')

        for i, interval in enumerate(result):
            assert index[i] in interval


class TestGetTimeBounds:
    """
    Tests for get_time_bounds function.
    """

    @pt.mark.parametrize('type_', ['series', 'data_array'])
    def test_raises_if_index_not_monotonic(
        self, type_: pd.Series | xr.DataArray
    ):
        """
        Should raise a ValueError if the index is not monotonic.
        """
        index = pd.DatetimeIndex(['2000-01-02', '2000-01-03', '2000-01-01'])

        if type_ == 'series':
            index = index.to_series()
        else:
            index = index.to_series().to_xarray()

        with pt.raises(ValueError):
            get_time_bounds(index)  # type: ignore

    @pt.mark.parametrize('type_', ['series', 'data_array'])
    @hp.given(index=datetime_indexes(min_size=3))
    def test_returns_correct_bounds(self, index: pd.DatetimeIndex, type_: str):
        """
        Should return the correct time bounds for a datetime index.
        """
        if type_ == 'series':
            data = index.to_series()
        else:
            data = index.to_series().to_xarray()

        bounds = get_time_bounds(data)

        assert bounds[0] == index.min().strftime('%Y-%m-%d')  # type: ignore
        assert bounds[1] == index.max().strftime('%Y-%m-%d')  # type: ignore
