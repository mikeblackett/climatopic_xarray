"""
climatopic_xarray.frequencies
========================
This module provides functions and classes for handling time frequencies in
xarray and pandas. It includes functions for inferring, parsing, and
converting time frequencies, as well as for converting datetime indices to
interval indices.
"""

import calendar
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    'Frequency',
    'get_freq',
    'get_offset',
    'get_period_alias',
    'get_time_bounds',
    'datetime_to_interval',
    'infer_freq',
    'infer_midpoint_freq',
    'get_time_bounds',
    'to_datetime_index',
    'ParsedFreq',
    'parse_freq',
]


class Frequency(StrEnum):
    """Valid frequencies"""

    DAY = 'D'
    MONTH = 'MS'
    SEASON = 'QS-DEC'
    YEAR = 'YS'


DAY_ANCHORS = [d.upper() for d in calendar.day_abbr if d]
"""Day anchors.

3-letter day-of-week abbreviations used in pandas frequency strings.
"""

MONTH_ANCHORS = [m.upper() for m in calendar.month_abbr if m]
"""List of Month anchors.

3-letter month-of-year abbreviations used in pandas frequency strings.
"""

OFFSET_TO_PERIOD_ALIAS: dict[str, str] = {
    'D': 'D',
    'W': 'W',
    'MS': 'M',
    'ME': 'M',
    'QS': 'Q',
    'QE': 'Q',
    'YS': 'Y',
    'YE': 'Y',
}
"""Period aliases for offset aliases.

Mapping from offset aliases to period aliases.
"""

OFFSET_END_TO_OFFSET_BEGIN: dict[str, str] = {
    'D': 'D',
    'W': 'W',
    'ME': 'MS',
    'QE': 'QS',
    'YE': 'YS',
}

for i, d in enumerate(DAY_ANCHORS):
    n = len(DAY_ANCHORS)
    ii = i + n % n - 1
    OFFSET_TO_PERIOD_ALIAS[f'W-{d}'] = f'W-{DAY_ANCHORS[ii]}'
    OFFSET_END_TO_OFFSET_BEGIN[f'W-{DAY_ANCHORS[ii]}'] = f'W-{d}'

for i, m in enumerate(MONTH_ANCHORS):
    n = len(MONTH_ANCHORS)
    ii = i + n % n - 1
    mm = MONTH_ANCHORS[ii]
    OFFSET_TO_PERIOD_ALIAS[f'QS-{m}'] = f'Q-{mm}'
    OFFSET_TO_PERIOD_ALIAS[f'YS-{m}'] = f'Y-{mm}'
    OFFSET_TO_PERIOD_ALIAS[f'QE-{m}'] = f'Q-{m}'
    OFFSET_TO_PERIOD_ALIAS[f'YE-{m}'] = f'Y-{m}'
    OFFSET_END_TO_OFFSET_BEGIN[f'QE-{mm}'] = f'QS-{m}'
    OFFSET_END_TO_OFFSET_BEGIN[f'YE-{mm}'] = f'YS-{m}'


OFFSET_TO_PERIOD_ALIAS = {
    k: OFFSET_TO_PERIOD_ALIAS[k] for k in sorted(OFFSET_TO_PERIOD_ALIAS)
}


def infer_midpoint_freq(
    index: pd.DatetimeIndex | pd.Series | xr.DataArray,
    how: Literal['start', 'end'] = 'start',
) -> str | None:
    """
    Infer the frequency of a datetime index by comparing the differences
    between consecutive elements.

    Parameters
    ----------
    index : pd.DatetimeIndex | pd.Series | xr.DataArray
        The index to infer the frequency from.
    how: Literal['start', 'end'], default 'start'
        Whether to align the inferred frequency to the start or the end of the
        period.

    Returns
    -------
    str | None
        The inferred frequency string or None if the frequency cannot be inferred.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If the index contains too few elements to infer a frequency or if the
        index is not 1D.

    See Also
    --------
    infer_freq : Infer the frequency of a time index
    """
    if isinstance(index, (xr.DataArray, pd.Series)):  # pragma: no cover
        index = to_datetime_index(index)

    if index.size < 4:
        # Pandas needs >= 3 elements, but we lose 1 when calculating the diffs
        raise ValueError(
            'An index must have at least 4 values to infer frequency '
            f'using the midpoint method; got {index.size=!r}'
        )

    if pd.infer_freq(index) == 'D':  # pragma: no cover
        return 'D'

    delta = index.diff()
    assert isinstance(delta, pd.TimedeltaIndex)
    left = (index - delta / 2).dropna()

    if freq := pd.infer_freq(left):
        if parse_freq(freq).base in ['D', 'W', 'M', 'Q', 'Y']:
            return freq
    # The remaining possibilities are 'MS', 'ME', 'QS', 'QE', 'YS', 'YE',
    # which are all 'multiples' of monthly frequencies, so we can snap to the
    # monthly frequency and infer the frequency from there.
    snap_freq = 'MS' if how == 'start' else 'ME'
    snapped = left.snap(snap_freq).normalize()
    return pd.infer_freq(snapped)


def infer_freq(
    index: pd.DatetimeIndex | xr.DataArray | pd.Series,
) -> str | None:
    """
    Return the most likely frequency of a pandas or xarray data object.

    This method first tries to infer the frequency using xarray's `infer_freq`
    method. If that fails, or if the inferred frequency is sub-daily, it falls
    back to the `infer_midpoint_freq` method.

    Parameters
    ----------
    index : pd.DatetimeIndex | xr.DataArray | pd.Series
        The index to infer the frequency from. If passed a Series or a
        DataArray, it will use the values of the object NOT the index.

    Returns
    -------
    str | None
        The frequency string or None if the frequency cannot be inferred.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If the index has too few elements to infer a frequency or the index
        is not 1D.

    See Also
    --------
    infer_midpoint_freq : Infer the frequency of a time index of midpoints.
    """
    freq = xr.infer_freq(index)
    if freq and freq.endswith('h'):
        # Some midpoint frequencies are inferred as 'h' (hourly) frequencies,
        # we consider this a failure.
        freq = None
    if freq is None:
        freq = infer_midpoint_freq(index)
    return freq


def get_freq(
    index: pd.DatetimeIndex | xr.DataArray | pd.Series,
) -> str:
    """
    Return the frequency of a pandas or xarray data object.

    This method first checks if the index has a frequency attribute. If not,
    it tries to infer the frequency using xarray's `infer_freq` method. If that
    fails, it falls back to the `infer_midpoint_freq` method. If all methods
    fail, it raises a ValueError.

    Parameters
    ----------
    index : pd.DatetimeIndex | xr.DataArray | pd.Series
        The index to infer the frequency from.

    Returns
    -------
    str
        The frequency string.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If the index has too few elements to infer a frequency or the index
        is not 1D.
    ValueError
        If a frequency cannot be inferred.

    See Also
    --------
    infer_freq : Infer the frequency of a time index
    infer_midpoint_freq : Infer the frequency of a time index of midpoints.
    """
    if isinstance(index, pd.DatetimeIndex) and index.freqstr:
        return index.freqstr
    freq = infer_freq(index)
    if freq is None:
        raise ValueError('Could not infer a frequency from the index.')
    return freq


def get_offset(freq: str) -> pd.DateOffset:
    """
    Return a DateOffset object from a frequency string.

    This is a wrapper around `pd.tseries.frequencies.to_offset` that raises a
    ValueError if the frequency string is invalid. Frequency strings that
    produce FutureWarnings are considered invalid.

    Parameters
    ----------
    freq : str
        The frequency string to convert to a DateOffset object.

    Returns
    -------
    pd.DateOffset
        A DateOffset object.

    Raises
    ------
    ValueError
        If the frequency string is invalid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='error', category=FutureWarning)
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except (FutureWarning, ValueError) as error:
            raise ValueError(f'Invalid frequency string: {freq=!r}') from error
    return cast(pd.DateOffset, offset)


def get_period_alias(freq: str) -> str:
    """
    Return the period alias for an offset alias.

    This implementation differs from the native pandas implementation in the
    way it handles anchored frequencies!

    Parameters
    ----------
    freq : str
        The offset alias to convert to a period alias.

    Returns
    -------
    str
        The period alias.

    Raises
    ------
    ValueError
        If the offset alias is invalid.
    """
    try:
        # Catch invalid freqs
        offset = get_offset(freq)
        # Catch unsupported freqs
        period_alias = OFFSET_TO_PERIOD_ALIAS[offset.base.freqstr]
    except (ValueError, KeyError) as error:
        raise ValueError(f'Invalid frequency: {freq}') from error

    if offset.n < 0:
        raise ValueError(
            f'frequency must be positive to represent a period: {freq}'
        )

    if offset.n > 1:
        period_alias = f'{offset.n}{period_alias}'
    return period_alias


def datetime_to_interval(
    index: pd.DatetimeIndex,
    *,
    label: Literal['left', 'middle', 'right'] | None = None,
    closed: Literal['left', 'right'] | None = None,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the bounds of a datetime index.

    The intervals represent the bins that would produce this index when using
    a pandas/xarray `resample` with the same `label` and `closed` arguments.
    Passing `'middle'` to the `label` parameter will coerce the index
    to its inferred frequency (see `infer_midpoint_freq` for details) and is
    equivalent to passing `label = None`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The datetime index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
        The default is ‘left’ for all frequency offsets except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which all have a default of ‘right’.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.
        The default is ‘left’ for all frequency offsets except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which all have a default of ‘right’.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.

    Raises
    ------
    TypeError
        If the index is not a datetime index.
    ValueError
        If a regular frequency cannot be inferred from the index.
    ValueError
        If the passed frequency does not conform to the index frequency.

    See Also
    --------
    infer_midpoint_freq : Infer the frequency of an index of time midpoints.
    """
    try:
        freq = get_freq(index)
    except ValueError as error:
        raise ValueError(
            'To convert a datetime index to an interval index, '
            'the index must have a regular frequency.'
        ) from error

    # Follow default parameters for pandas/xarray `resample`.
    is_end_aligned = freq in {'ME', 'YE', 'QE', 'W'}
    if label is None:
        label = 'right' if is_end_aligned else 'left'
    if closed is None:
        closed = 'right' if is_end_aligned else 'left'

    if label == 'middle':
        # Coerce the midpoints to their inferred frequency
        index = cast(
            pd.DatetimeIndex,
            index.to_series().asfreq(freq).index.normalize(),  # pyright: ignore [reportAttributeAccessIssue]
        )
        index = index.union([index[-1] + index.freq]).shift(-1)

    if label == 'right':
        left = index.shift(periods=-1, freq=freq)
    else:
        left = index

    right = left.shift(periods=1, freq=freq)

    return pd.IntervalIndex.from_arrays(
        left=left,
        right=right,
        closed=cast(Literal['left', 'right'], closed),
    )


def get_time_bounds(
    data: xr.DataArray | pd.Series,
) -> tuple[str, str]:
    """
    Return the minimum and maximum time values of the xarray object.

    This method expects the object to have time coordinate with a
    monotonic-increasing datetime-like index.

    Parameters
    ----------
    data : DataArray | Dataset
        The xarray object to calculate the climatology. It must have a
        datetime-like index.

    Returns
    -------
    tuple[str, str]
        A tuple containing the minimum and maximum time values as strings.
    """
    index = to_datetime_index(data)
    if not any([index.is_monotonic_increasing, index.is_monotonic_decreasing]):
        raise ValueError('Expected data to be monotonic.')
    bounds = index[0 :: index.size - 1].strftime('%Y-%m-%d').tolist()
    return bounds[0], bounds[1]


@dataclass(frozen=True)
class ParsedFreq:
    """
    A parsed pandas frequency string.

    Attributes
    ----------
    base : str
        The base frequency alias.
    n : int, optional
        The multiplier for the base frequency, default is 1.
    boundary : Literal['S', 'E'] | None, optional
        The boundary for the frequency, default is None.
    anchor : str | None, optional
        The anchor for the frequency, default is None.
    """

    base: str
    n: int = 1
    boundary: Literal['S', 'E'] | None = None
    anchor: str | None = None


def parse_freq(freq: str) -> ParsedFreq:
    """
    Parse a pandas frequency string into its components.

    Parameters
    ----------
    freq : str
        The frequency string to parse

    Returns
    -------
    ParsedFreq
        A ParsedFreq object

    Raises
    ------
    ValueError
        If the freq is invalid

    See Also
    --------
    https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    """
    offset = get_offset(freq)
    boundary = None
    base, *rest = offset.name.split(sep='-')
    if base.endswith(('S', 'E')):
        boundary = cast(Literal['S', 'E'], base[-1])
        base = base[:-1]
    anchor = rest[0] if rest else None
    return ParsedFreq(
        n=offset.n,
        base=base,
        boundary=boundary,
        anchor=anchor,
    )


def to_datetime_index(
    data: xr.DataArray | pd.Series | np.ndarray,
) -> pd.DatetimeIndex:
    """Convert a 1D xarray or pandas object to a DatetimeIndex."""
    if data.ndim != 1:
        raise ValueError(f'Index must be 1D; got {data.ndim=!r}')
    if not np.issubdtype(
        xr.Variable(dims='dim', data=data).dtype, np.datetime64
    ):
        raise ValueError(f"'data' must be datetime-like; got {data.dtype=!r}")
    values = data if isinstance(data, np.ndarray) else data.values
    return pd.DatetimeIndex(values)
