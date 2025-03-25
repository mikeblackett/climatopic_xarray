import calendar
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

import hypothesis as hp
import hypothesis.strategies as st
import pandas as pd
from hypothesis.errors import InvalidArgument
from pandas.core.tools.datetimes import DatetimeScalar
from pandas.errors import OutOfBoundsDatetime

from climatopic_xarray.testing.strategies._offset_aliases import (
    OffsetCategories,
    query_offset_strings,
)


type TimeUnitT = Literal['D', 'h', 'min', 's', 'ms', 'us', 'ns']

MIN_TIMESTAMP = pd.Timestamp.min.ceil('ms')
MAX_TIMESTAMP = pd.Timestamp.max.floor('ms')
MONTH_ABBR = list(calendar.month_abbr)
MONTH_NAME = list(calendar.month_name)


class _TimeUnit(StrEnum):
    DAY = 'D'
    HOUR = 'h'
    MINUTE = 'min'
    SECOND = 's'
    MILLISECOND = 'ms'
    MICROSECOND = 'us'
    NANOSECOND = 'ns'


@st.composite
def offset_aliases(
    draw: st.DrawFn,
    *,
    categories: OffsetCategories | None = None,
    exclude_categories: OffsetCategories | None = None,
    exclude_freqs: Iterable[str] | None = None,
    include_freqs: Iterable[str] | None = None,
    min_n: int | None = None,
    max_n: int | None = None,
) -> str:
    """
    Generate pandas offset aliases.

    Parameters
    ----------
    categories : OffsetCategories | None, optional
        The categories to use, by default None.
        If None, all categories are used.
    exclude_categories : OffsetCategories | None, optional
        The categories to exclude, by default None.
        If None, no categories are excluded.
    exclude_freqs : Iterable[str] | None, optional
        The frequencies to exclude, by default None.
        These frequencies will always be excluded from the generated frequencies.
    include_freqs : Iterable[str] | None, optional
        The frequencies to include, by default None.
        These frequencies will always be included in the generated frequencies.
    min_n : int | None, optional
        The minimum frequency multiplier (`n` in pandas offsets), by default None.
        If None, the minimum is 1.
    max_n : int | None, optional
        The maximum frequency multiplier (`n` in pandas offsets), by default None.
        If None, the maximum is 3.

    Returns
    -------
    str
        A pandas offset alias.

    """
    freqs = query_offset_strings(
        categories=categories,
        exclude_categories=exclude_categories,
        exclude_freqs=exclude_freqs,
        include_freqs=include_freqs,
    )
    freq = draw(st.sampled_from(freqs))

    min_n = 1 if min_n is None else min_n
    max_n = 3 if max_n is None else max_n
    if min_n > max_n:
        raise InvalidArgument(f'{min_n=!r} cannot be greater than {max_n=!r}.')
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    hp.assume(n != 0)

    if n == 1 and draw(st.booleans()):
        # 1MS and MS are equivalent
        return freq
    return f'{n}{freq}'


@st.composite
def timestamps(
    draw: st.DrawFn,
    *,
    min_value: DatetimeScalar | None = None,
    max_value: DatetimeScalar | None = None,
    normalize: bool | None = None,
    unit: TimeUnitT | None = None,
) -> pd.Timestamp:
    """
    Generate pandas timestamps.

    Parameters
    ----------
    min_value : DatetimeScalar | None, optional
        The minimum value of the timestamp, by default None.
        If None, `pandas.Timestamp.min` is used.
    max_value : DatetimeScalar | None, optional
        The maximum value of the timestamp, by default None.
        If None, `pandas.Timestamp.max` is used.
    unit : TimeUnitT | None, optional
        The unit of the timestamp, by default None.
        If None, a random unit is used.
    normalize : bool | None, optional
        Whether to normalize the timestamp, by default None.
        If None, normalization is random.


    Returns
    -------
    pd.Timestamp
        A pandas timestamp.
    """
    # TODO: add support for timezone-aware timestamps
    try:
        min_timestamp = pd.to_datetime(min_value or MIN_TIMESTAMP)
        max_timestamp = pd.to_datetime(max_value or MAX_TIMESTAMP)
    except (ValueError, TypeError) as error:
        raise InvalidArgument(f'Invalid timestamp: {error}')

    datetime = draw(
        st.datetimes(
            min_value=min_timestamp.to_pydatetime(False),
            max_value=max_timestamp.to_pydatetime(False),
        )
    )

    freq = draw(st.sampled_from(_TimeUnit)) if unit is None else unit
    # If the timestamp is close to the min or max timestamp, rounding could
    # result in an out-of-bounds timestamp.
    try:
        timestamp = pd.to_datetime(datetime).floor(freq)
    except OutOfBoundsDatetime:
        timestamp = pd.to_datetime(datetime).ceil(freq)

    normalize = draw(st.booleans()) if normalize is None else normalize
    if normalize:
        timestamp = timestamp.normalize()

    return timestamp


@st.composite
def datetime_indexes(
    draw: st.DrawFn,
    *,
    freqs: st.SearchStrategy[str] | None = None,
    min_size: int = 0,
    max_size: int | None = None,
    name: st.SearchStrategy[str] | None = None,
    normalize: bool | None = None,
) -> pd.DatetimeIndex:
    """
    Generate pandas datetime indexes with regular frequencies.

    Parameters
    ----------
    freqs : st.SearchStrategy[str] | None, optional
        The frequency strings to use, by default None.
        If None, frequency strings are generated randomly.
    min_size : int, optional
        The minimum size of the index, by default 0.
    max_size : int, optional
        The maximum size of the index, by default 10.
    name : st.SearchStrategy[str] | None, optional
        The name of the index, by default None.
        If None, a random string is used.
    normalize : bool | None, optional
        Whether to normalize the timestamps, by default None.
        If None, normalization is random.

    Returns
    -------
    pd.DatetimeIndex
        A pandas datetime index with a regular frequency.
    """
    if min_size < 0:
        raise InvalidArgument(f'Cannot have {min_size=!r} < 0')
    if max_size is None:
        max_size = draw(st.integers(min_value=min_size, max_value=100))
    if max_size and max_size < min_size:
        raise InvalidArgument(f'cannot have {min_size=!r} > {max_size=!r}')

    timestamp = draw(timestamps(normalize=normalize))
    freq = draw(offset_aliases() if freqs is None else freqs)
    periods = draw(st.integers(min_value=min_size, max_value=max_size))
    label = draw(st.text() if name is None else name)
    kwargs = {
        'periods': periods,
        'freq': freq,
        'name': label,
    }

    # Depending on the timestamp and periods, the index may go out of bounds.
    try:
        # try going forwards ...
        index = pd.date_range(start=timestamp, **kwargs)
    except OutOfBoundsDatetime:
        # ... then try going backwards
        index = pd.date_range(end=timestamp, **kwargs)
    return index


@st.composite
def month_ordinals(
    draw: st.DrawFn, min_value: int | None = None, max_value: int | None = None
) -> calendar.Month:
    """
    Generate month-of-year ordinals.

    Parameters
    ----------
    min_value : int | None, optional
        The minimum value of the month-of-year ordinal, by default None.
    max_value : int | None, optional
        The maximum value of the month-of-year ordinal, by default None.

    Returns
    -------
    calendar.Month
        A month-of-year ordinal.
    """
    # calendar.Month is 1-indexed!!!
    min_value = 1 if min_value is None else min_value
    max_value = 12 if max_value is None else max_value

    if min_value < 1:
        raise InvalidArgument(f'cannot have min_value={min_value} < 1')
    if max_value > 12:
        raise InvalidArgument(f'cannot have max_value={max_value} > 12')

    ordinal = draw(st.integers(min_value=min_value, max_value=max_value))
    return calendar.Month(ordinal)


@st.composite
def month_names(
    draw: st.DrawFn, min_value: str | None = None, max_value: str | None = None
) -> str:
    """
    Generate month-of-year names.

    Equality is determined using the ordinal position of each month of the year.

    Parameters
    ----------
    min_value : str | None, optional
        The minimum value of the month-of-year name, by default None.
    max_value : str | None, optional
        The maximum value of the month-of-year name, by default None.

    Returns
    -------
    str
        A localized month-of-year name.
    """
    # MONTH_NAME is 1-indexed!!!
    try:
        min_ordinal = MONTH_NAME.index(min_value or MONTH_NAME[1])
        max_ordinal = MONTH_NAME.index(max_value or MONTH_NAME[-1])
    except ValueError:
        raise InvalidArgument(
            f'invalid month name: {min_value=}, {max_value=}'
        )
    ordinal = draw(
        month_ordinals(min_value=min_ordinal, max_value=max_ordinal)
    )
    return calendar.month_name[ordinal]


@st.composite
def month_abbreviations(
    draw: st.DrawFn, min_value: str | None = None, max_value: str | None = None
) -> str:
    """
    Generate month-of-year abbreviations.

    Equality is determined using the ordinal position of each month of the year.

    Parameters
    ----------
    min_value : str | None, optional
        The minimum value of the month-of-year abbreviation, by default None.
    max_value : str | None, optional
        The maximum value of the month-of-year abbreviation, by default None.

    Returns
    -------
    str
        A localized month-of-year abbreviation.
    """
    # MONTH_ABBR is 1-indexed!!!
    min_ordinal = MONTH_ABBR.index(min_value or MONTH_ABBR[1])
    max_ordinal = MONTH_ABBR.index(max_value or MONTH_ABBR[-1])
    ordinal = draw(
        month_ordinals(min_value=min_ordinal, max_value=max_ordinal)
    )
    return calendar.month_abbr[ordinal]
